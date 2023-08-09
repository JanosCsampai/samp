import itertools
import copy

import numpy as np
import torch

import trajnetplusplustools

from .modules import Hidden2Normal, InputEmbedding, Hidden2Force, Hidden2ForceField

from .. import augmentation
from .utils import center_scene

NAN = float('nan')

def drop_distant(xy, r=6.0):
    """
    Drops pedestrians more than r meters away from primary ped
    """
    distance_2 = np.sum(np.square(xy - xy[:, 0:1]), axis=2)
    mask = np.nanmin(distance_2, axis=0) < r**2
    return xy[:, mask], mask


def generate_pooling_inputs(obs2, obs1, hidden_cell_state, track_mask, batch_split):
    hidden_states_to_pool = torch.stack(hidden_cell_state[0]).clone() # detach?
    hidden_dim = hidden_states_to_pool.size(-1)
    # tensor for pooling; filled with nan-mask [bs, max # neighbor, 2]
    max_num_neighbor = (batch_split[1:] - batch_split[:-1]).max()   # number of agents in a scene minus the primary
    batch_size = len(batch_split) - 1
    curr_positions = torch.empty(batch_size, max_num_neighbor, 2).fill_(float('nan')).to(obs1.device) # placeholder
    prev_positions = torch.empty(batch_size, max_num_neighbor, 2).fill_(float('nan')).to(obs1.device) # placeholder
    curr_hidden_state = torch.empty(batch_size, max_num_neighbor, hidden_dim).fill_(float('nan')).to(obs1.device) # placeholder
    track_mask_positions = torch.empty(batch_size, max_num_neighbor).fill_(False).bool().to(obs1.device)  # placeholder

    for i in range(batch_size):
        curr_positions[i, :batch_split[i+1]-batch_split[i]] = obs2[batch_split[i]:batch_split[i+1]]
        prev_positions[i, :batch_split[i+1]-batch_split[i]] = obs1[batch_split[i]:batch_split[i+1]]
        curr_hidden_state[i, :batch_split[i+1]-batch_split[i]] = hidden_states_to_pool[batch_split[i]:batch_split[i+1]]
        track_mask_positions[i, :batch_split[i+1]-batch_split[i]] = track_mask[batch_split[i]:batch_split[i+1]].bool()

    return curr_positions, prev_positions, curr_hidden_state, track_mask_positions


class LSTM(torch.nn.Module):
    def __init__(self, embedding_dim=64, hidden_dim=128, pool=None, pool_to_input=True, goal_dim=None, goal_flag=False, force_field_resolution=12, plot_name="", is_highway=False):
        """ Initialize the LSTM forecasting model

        Attributes
        ----------
        embedding_dim : Embedding dimension of location coordinates
        hidden_dim : Dimension of hidden state of LSTM
        pool : interaction module
        pool_to_input : Bool
            if True, the interaction vector is concatenated to the input embedding of LSTM [preferred]
            if False, the interaction vector is added to the LSTM hidden-state
        goal_dim : Embedding dimension of the unit vector pointing towards the goal
        goal_flag: Bool
            if True, the embedded goal vector is concatenated to the input embedding of LSTM 
        """

        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.pool = pool
        self.pool_to_input = pool_to_input

        self.force_field_resolution = force_field_resolution
        self.plot_vector_field_freq = 500
        self.plot_counter = 0
        self.plot_counter_check = self.plot_counter + 1
        self.plot_name = plot_name
        self.is_highway = is_highway

        scale = 4.0
        self.input_embedding = InputEmbedding(2, self.embedding_dim, scale)

        ## Goal
        self.goal_flag = goal_flag
        self.goal_dim = goal_dim or embedding_dim
        self.goal_embedding = InputEmbedding(2, self.goal_dim, scale)
        goal_rep_dim = self.goal_dim if self.goal_flag else 0

        ## Pooling
        pooling_dim = 0
        if pool is not None and self.pool_to_input:
            pooling_dim = self.pool.out_dim 
        
        ## LSTMs
        self.encoder = torch.nn.LSTMCell(self.embedding_dim + goal_rep_dim + pooling_dim, self.hidden_dim)
        self.decoder = torch.nn.LSTMCell(self.embedding_dim + goal_rep_dim + pooling_dim, self.hidden_dim)

        # Predict the parameters of a multivariate normal:
        # mu_vel_x, mu_vel_y, sigma_vel_x, sigma_vel_y, rho
        self.hidden2normal = Hidden2Normal(self.hidden_dim)
        self.hidden2force = Hidden2Force(self.hidden_dim)
        self.hidden2ForceField = Hidden2ForceField(self.hidden_dim, self.force_field_resolution*self.force_field_resolution*2)

    def step(self, lstm, hidden_cell_state, obs1, obs2, goals, batch_split):
        """Do one step of prediction: two inputs to one normal prediction.
        
        Parameters
        ----------
        lstm: torch nn module [Encoder / Decoder]
            The module responsible for prediction
        hidden_cell_state : tuple (hidden_state, cell_state)
            Current hidden_cell_state of the pedestrians
        obs1 : Tensor [num_tracks, 2]
            Previous x-y positions of the pedestrians
        obs2 : Tensor [num_tracks, 2]
            Current x-y positions of the pedestrians
        goals : Tensor [num_tracks, 2]
            Goal coordinates of the pedestrians
        
        Returns
        -------
        hidden_cell_state : tuple (hidden_state, cell_state)
            Updated hidden_cell_state of the pedestrians
        normals : Tensor [num_tracks, 5]
            Parameters of a multivariate normal of the predicted position 
            with respect to the current position
        """
        num_tracks = len(obs2)
        # mask for pedestrians absent from scene (partial trajectories)
        # consider only the hidden states of pedestrains present in scene
        track_mask = (torch.isnan(obs1[:, 0]) + torch.isnan(obs2[:, 0])) == 0

        ## Masked Hidden Cell State
        hidden_cell_stacked = [
            torch.stack([h for m, h in zip(track_mask, hidden_cell_state[0]) if m], dim=0),
            torch.stack([c for m, c in zip(track_mask, hidden_cell_state[1]) if m], dim=0),
        ]

        ## Mask current velocity & embed
        curr_velocity = obs2 - obs1
        curr_velocity = curr_velocity[track_mask]
        input_emb = self.input_embedding(curr_velocity)

        ## Mask Goal direction & embed
        if self.goal_flag:
            ## Get relative direction to goals (wrt current position)
            norm_factors = (torch.norm(obs2 - goals, dim=1))
            goal_direction = (obs2 - goals) / norm_factors.unsqueeze(1)
            goal_direction[norm_factors == 0] = torch.tensor([0., 0.], device=obs1.device)
            goal_direction = goal_direction[track_mask]
            goal_emb = self.goal_embedding(goal_direction)
            input_emb = torch.cat([input_emb, goal_emb], dim=1)

        ## Mask & Pool per scene
        if self.pool is not None:
            curr_positions, prev_positions, curr_hidden_state, track_mask_positions = \
                generate_pooling_inputs(obs2, obs1, hidden_cell_state, track_mask, batch_split)
            pool_sample = self.pool(curr_hidden_state, prev_positions, curr_positions)
            pooled = pool_sample[track_mask_positions.view(-1)]

            if self.pool_to_input:
                input_emb = torch.cat([input_emb, pooled], dim=1)
            else:
                hidden_cell_stacked[0] += pooled

        # LSTM step
        hidden_cell_stacked = lstm(input_emb, hidden_cell_stacked)
        normal_masked = self.hidden2normal(hidden_cell_stacked[0])

        # unmask [Update hidden-states and next velocities of pedestrians]
        normal = torch.full((track_mask.size(0), 5), NAN, device=obs1.device)
        mask_index = [i for i, m in enumerate(track_mask) if m]
        for i, h, c, n in zip(mask_index,
                              hidden_cell_stacked[0],
                              hidden_cell_stacked[1],
                              normal_masked):
            hidden_cell_state[0][i] = h
            hidden_cell_state[1][i] = c
            normal[i] = n

        return hidden_cell_state, normal

    def forward(self, observed, goals, batch_split, prediction_truth=None, n_predict=None, plot_name=""):
        """Forecast the entire sequence 
        
        Parameters
        ----------
        observed : Tensor [obs_length, num_tracks, 2]
            Observed sequences of x-y coordinates of the pedestrians
        goals : Tensor [num_tracks, 2]
            Goal coordinates of the pedestrians
        batch_split : Tensor [batch_size + 1]
            Tensor defining the split of the batch.
            Required to identify the tracks of to the same scene        
        prediction_truth : Tensor [pred_length - 1, num_tracks, 2]
            Prediction sequences of x-y coordinates of the pedestrians
            Helps in teacher forcing wrt neighbours positions during training
        n_predict: Int
            Length of sequence to be predicted during test time

        Returns
        -------
        rel_pred_scene : Tensor [pred_length, num_tracks, 5]
            Predicted velocities of pedestrians as multivariate normal
            i.e. positions relative to previous positions
        pred_scene : Tensor [pred_length, num_tracks, 2]
            Predicted positions of pedestrians i.e. absolute positions
        """
        assert ((prediction_truth is None) + (n_predict is None)) == 1
        if n_predict is not None:
            # -1 because one prediction is done by the encoder already
            prediction_truth = [None for _ in range(n_predict - 1)]

        # initialize: Because of tracks with different lengths and the masked
        # update, the hidden state for every LSTM needs to be a separate object
        # in the backprop graph. Therefore: list of hidden states instead of
        # a single higher rank Tensor.
        num_tracks = observed.size(1)
        hidden_cell_state = (
            [torch.zeros(self.hidden_dim, device=observed.device) for _ in range(num_tracks)],
            [torch.zeros(self.hidden_dim, device=observed.device) for _ in range(num_tracks)],
        )

        ## Reset LSTMs of Interaction Encoders.
        if self.pool is not None:
            max_num_neighbor = (batch_split[1:] - batch_split[:-1]).max() - 1
            batch_size = len(batch_split) - 1
            self.pool.reset(batch_size * (max_num_neighbor+1), max_num_neighbor, device=observed.device)

        # list of predictions
        normals = []  # predicted normal parameters for both phases
        positions = []  # true (during obs phase) and predicted positions

        if len(observed) == 2:
            positions = [observed[-1]]

        # encoder
        for obs1, obs2 in zip(observed[:-1], observed[1:]):
            ##LSTM Step
            hidden_cell_state, normal = self.step(self.encoder, hidden_cell_state, obs1, obs2, goals, batch_split)

            #velocity = self.encoder_to_force_field(normal, obs1, obs2, batch_split, observed)

            # concat predictions
            normals.append(normal)
            positions.append(obs2 + normal[:, :2])  # no sampling, just mean

        # initialize predictions with last position to form velocity. DEEP COPY !!!
        prediction_truth = copy.deepcopy(list(itertools.chain.from_iterable(
            (observed[-1:], prediction_truth)
        )))

        # decoder, predictions
        for obs1, obs2 in zip(prediction_truth[:-1], prediction_truth[1:]):
            if obs1 is None:
                obs1 = positions[-2].detach()  # DETACH!!!
            else:
                for primary_id in batch_split[:-1]:
                    obs1[primary_id] = positions[-2][primary_id].detach()  # DETACH!!!
            if obs2 is None:
                obs2 = positions[-1].detach()
            else:
                for primary_id in batch_split[:-1]:
                    obs2[primary_id] = positions[-1][primary_id].detach()  # DETACH!!!
            hidden_cell_state, normal = self.step(self.decoder, hidden_cell_state, obs1, obs2, goals, batch_split)

            #velocity = self.encoder_to_force_field(normal, obs1, obs2, batch_split, observed=observed)

            # concat predictions
            normals.append(normal)
            positions.append(obs2 + normal[:, :2]) # no sampling, just mean

        # Pred_scene: Tensor [seq_length, num_tracks, 2]
        #    Absolute positions of all pedestrians
        # Rel_pred_scene: Tensor [seq_length, num_tracks, 5]
        #    Velocities of all pedestrians
        rel_pred_scene = torch.stack(normals, dim=0)
        pred_scene = torch.stack(positions, dim=0)

        return rel_pred_scene, pred_scene
    
    
    def encoder_to_force_field(self, force_field, obs1, obs2, batch_split, observed, positions, step_index, save_plots=False):
        # Compute velocities
        velocities = obs2 - obs1

        # Set device to GPU
        device = torch.device("cuda:1")

        # Preallocate memory for forces
        forces = torch.zeros((len(obs1), 2), device=device)

        # Define scene boundaries
        x_min_scene, x_max_scene = (-4, 4)
        y_min_scene, y_max_scene = (-4, 4) if not self.is_highway else (-1, 1)

        # Reshape the force field for convenient indexing
        decoder_output_reshaped = force_field.view(len(velocities), self.force_field_resolution, self.force_field_resolution, 2)

        # Create a grid of points within the scene
        x_scene, y_scene = torch.linspace(x_min_scene, x_max_scene, self.force_field_resolution, device=device), torch.linspace(y_min_scene, y_max_scene, self.force_field_resolution, device=device)
        X_scene, Y_scene = torch.meshgrid(x_scene, y_scene)
        frames = []

        # Loop over all scenes
        for _, (start, end) in enumerate(zip(batch_split[:-1], batch_split[1:])):
            # Extract current scene from encoder output
            obs1m_scene, obs2m_scene = obs1[start:end], obs2[start:end]
            #if (start == 0): print("First Scene: ", obs2m_scene)

            # Initialize zero force for current scene
            init_forces = torch.zeros(end-start, 2, device=device)

            # Create a vector field with only zero vectors
            U_scene = init_forces[:, 0].unsqueeze(-1).unsqueeze(-1) * torch.ones_like(X_scene)
            V_scene = init_forces[:, 1].unsqueeze(-1).unsqueeze(-1) * torch.ones_like(Y_scene)

            # Add decoder output to the vector field
            U_scene += decoder_output_reshaped[end-start if len(batch_split) != 2 else slice(None), : , : , 0]
            V_scene += decoder_output_reshaped[end-start if len(batch_split) != 2 else slice(None), : , : , 1]

            # Initialize a vector array for interpolation
            vectors = torch.zeros((end-start, 2, self.force_field_resolution, self.force_field_resolution), device=device)
            vectors[:, 0, :, :], vectors[:, 1, :, :] = U_scene, V_scene

            points = torch.stack([X_scene.flatten(), Y_scene.flatten()], dim=-1)

            # Normalize the pedestrian's current position to [-1, 1]
            current_positions = obs2m_scene.nan_to_num(0)
            current_positions[:, 0] = (((current_positions[:, 0] - x_min_scene) / (x_max_scene - x_min_scene)) * 2) - 1
            current_positions[:, 1] = (((current_positions[:, 1] - y_min_scene) / (y_max_scene - y_min_scene)) * 2) - 1

            # Reshape the current positions for interpolation
            current_positions = current_positions.view(end-start, 1, 1, 2)

            # Interpolate the force field to find the force at the pedestrian's current position
            interpolated = torch.nn.functional.grid_sample(vectors.nan_to_num(0), current_positions, padding_mode="zeros", align_corners=False, mode="bilinear")

            # interpolated = self.nearest_neighbor_interpolation(points, vectors, current_positions)

            # interpolated = closest_vector_interpolation(vectors.nan_to_num(0), current_positions.view(end - start, -1, 2))

            # Add the force to the force array
            if torch.isnan(points).any():
                forces[start:end] = torch.zeros((end - start, 2), device=device)
            else:
                forces[start:end] = interpolated.view(end-start, 2)
                i = start
                if save_plots:
                    self.plot_counter_check += 1
                    if (torch.isnan(obs1m_scene[i-start, 0])): continue
                    # Plot the vector field
                        # Create a new figure for the animation
                    #fig = plt.figure(figsize=(x_max_scene-x_min_scene+2, y_max_scene-y_min_scene+2))

                    # Create a list to store the frames of the animation
                    # Define the region around the main agent
                    agent_x = obs1m_scene[i-start, 0].detach().cpu()
                    agent_y = obs1m_scene[i-start, 1].detach().cpu()
                    radius = 1.0  # adjust this value to change the size of the region around the agent

                    # Create a mask for the region around the main agent
                    mask = ((X_scene.detach().cpu() - agent_x)**2 + 
                            (Y_scene.detach().cpu() - agent_y)**2
                        ) < radius**2

                    # Apply the mask to the vector field
                    U_masked = U_scene[i-start].detach().cpu()# * mask
                    V_masked = V_scene[i-start].detach().cpu()# * mask

                    # Plot the masked vector field
                    frames.append(plt.quiver(
                        X_scene.detach().cpu(), 
                        Y_scene.detach().cpu(), 
                        U_masked.cpu().detach(), 
                        V_masked.cpu().detach(), 
                        scale_units='xy', 
                        scale=1, 
                        color='0.3', 
                        width=0.005
                    ))
                    scale = 1
                    # Plot the trajectory of the pedestrian
                    frames.append(plt.plot(
                        [
                            obs1m_scene[i-start, 0] / scale, 
                            obs2m_scene[i-start, 0] / scale
                        ], 
                        [
                            obs1m_scene[i-start, 1] / scale, 
                            obs2m_scene[i-start, 1] / scale
                        ], 
                        'r-', linewidth=2, color="yellow"
                    )[0])
                    if positions:
                        # plot positions from observed sequence till now
                        for j in range(len(observed)-1, len(positions)):
                            frames.append(plt.arrow(
                                    positions[j-1][start][0].cpu().detach(), 
                                    positions[j-1][start][1].cpu().detach(), 
                                    positions[j][start][0].cpu().detach() - positions[j-1][start][0].cpu().detach(), 
                                    positions[j][start][1].cpu().detach() - positions[j-1][start][1].cpu().detach(), 
                                    width=0.05, color="yellow"))
                    if observed is not None:
                        for j in range(start, end):
                            past_positions = observed[:, j, :]
                            color = "blue"
                            if j == i: color = "red"
                            
                            for k in range(len(past_positions) - 1):
                                if (k == len(past_positions) - 2):
                                    frames.append(plt.arrow(
                                    past_positions[-2][0], 
                                    past_positions[-2][1], 
                                    past_positions[-1][0].cpu() - past_positions[-2][0].cpu(), 
                                    past_positions[-1][1].cpu() - past_positions[-2][1].cpu(), 
                                    width=0.05, color=color))
                                else:
                                    frames.append(plt.plot(
                                        [
                                            past_positions[k][0] / scale, 
                                            past_positions[k+1][0] / scale
                                        ], 
                                        [
                                            past_positions[k][1] / scale, 
                                            past_positions[k+1][1] / scale
                                        ], 
                                        'b-', linewidth=3, color=color
                                    )[0])
                                
                    force = interpolated[i - start]
                                
                    frames.append(plt.arrow(
                        obs2m_scene[i-start, 0].cpu().detach() / scale, 
                        obs2m_scene[i-start, 1].cpu().detach() / scale, 
                        force[0].cpu().detach() / scale, 
                        force[1].cpu().detach() / scale, 
                        color='pink', width=0.05
                    ))

                    
                    
                    
                    # plt.title(f'Vector field for pedestrian {i}')
                    # # Save the plot as an image file
                    # if not os.path.exists(f'intermediate-plots/{self.plot_name}'):
                    #     # If the directory does not exist, create it
                    #     os.makedirs(f'intermediate-plots/{self.plot_name}')
                    # plt.savefig(
                    #     f'intermediate-plots/{self.plot_name}/vector_field_scene_{self.plot_counter}_step_{step_index}_pedestrian_{i}.png'
                    # )
                    # plt.close()


        return forces, frames

class LSTMPredictor(object):
    def __init__(self, model):
        self.model = model

    def save(self, state, filename):
        with open(filename, 'wb') as f:
            torch.save(self, f)

        # # during development, good for compatibility across API changes:
        # # Save state for optimizer to continue training in future
        with open(filename + '.state', 'wb') as f:
            torch.save(state, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return torch.load(f)


    def __call__(self, paths, scene_goal, n_predict=12, modes=1, predict_all=True, obs_length=9, start_length=0, args=None):
        self.model.eval()
        # self.model.train()
        with torch.no_grad():
            xy = trajnetplusplustools.Reader.paths_to_xy(paths)
            # xy = augmentation.add_noise(xy, thresh=args.thresh, ped=args.ped_type)
            batch_split = [0, xy.shape[1]]

            if args.normalize_scene:
                xy, rotation, center, scene_goal = center_scene(xy, obs_length, goals=scene_goal)
            
            xy = torch.Tensor(xy)  #.to(self.device)
            scene_goal = torch.Tensor(scene_goal) #.to(device)
            batch_split = torch.Tensor(batch_split).long()

            multimodal_outputs = {}
            for num_p in range(modes):
                # _, output_scenes = self.model(xy[start_length:obs_length], scene_goal, batch_split, xy[obs_length:-1].clone())
                _, output_scenes = self.model(xy[start_length:obs_length], scene_goal, batch_split, n_predict=n_predict)
                output_scenes = output_scenes.numpy()
                if args.normalize_scene:
                    output_scenes = augmentation.inverse_scene(output_scenes, rotation, center)
                output_primary = output_scenes[-n_predict:, 0]
                output_neighs = output_scenes[-n_predict:, 1:]
                ## Dictionary of predictions. Each key corresponds to one mode
                multimodal_outputs[num_p] = [output_primary, output_neighs]

        ## Return Dictionary of predictions. Each key corresponds to one mode
        return multimodal_outputs