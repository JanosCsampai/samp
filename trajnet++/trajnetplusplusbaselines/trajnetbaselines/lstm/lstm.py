import itertools
import copy

import numpy as np
import torch

import trajnetplusplustools

import matplotlib.pyplot as plt

from .modules import Hidden2Normal, InputEmbedding, Hidden2ForceField


from scipy.interpolate import griddata

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
    def __init__(self, embedding_dim=64, hidden_dim=128, pool=None, pool_to_input=True, goal_dim=None, goal_flag=False, force_field_resolution=4):
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

        ## Location
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
        self.hidden2ForceField = Hidden2ForceField(self.hidden_dim, self.force_field_resolution*self.force_field_resolution*2)


    def plot_vector_field(self, obs1, obs2):
        """Plots a vector field diagram given multivariate normal parameters.

        Args:
            normals (Tensor): [num_tracks, 5] Parameters of multivariate normals.
            obs1 (Tensor): [num_tracks, 2] Previous positions of pedestrians.
            obs2 (Tensor): [num_tracks, 2] Current positions of pedestrians.
        """
        # Plot vector field
        obs1 = obs1.cpu().numpy()
        obs2 = obs2.cpu().numpy()
        obs1 = obs1[0]
        obs2 = obs2[0]
        plt.plot(obs1[:, 0], obs1[:, 1], 'bo', markersize=1, color="black")
        print(obs2[:, 0] - obs1[:, 0])

        # Plot arrows from previous to current positions
        plt.quiver(obs1[:, 0], obs1[:, 1], obs2[:, 0] - obs1[:, 0], 
                obs2[:, 1] - obs1[:, 1], color='r', angles='xy', 
                scale_units='xy', scale=40)

        # Set plot limits
        plt.xlim(np.min(obs1[:, 0]), np.max(obs1[:, 0]))
        plt.ylim(np.min(obs1[:, 1]), np.max(obs1[:, 1]))
        plt.savefig("vector_field.png")

    def step(self, lstm, hidden_cell_state, obs1, obs2, goals, batch_split, observed = None):
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
        if lstm == self.encoder:
            #full_force = torch.mul(hidden_cell_state[0][1], hidden_cell_state[1][1])
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
            force_field = self.hidden2ForceField(hidden_cell_stacked[0])

            # unmask [Update hidden-states and next velocities of pedestrians]
            normal = torch.full((track_mask.size(0), self.force_field_resolution*self.force_field_resolution*2), NAN, device=obs1.device)
            mask_index = [i for i, m in enumerate(track_mask) if m]
            for i, h, c, n in zip(mask_index,
                                hidden_cell_stacked[0],
                                hidden_cell_stacked[1],
                                force_field):
                hidden_cell_state[0][i] = h
                hidden_cell_state[1][i] = c
                normal[i] = n

            return hidden_cell_state, normal

    def forward(self, observed, goals, batch_split, prediction_truth=None, n_predict=None):
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
        print("lstm.forward")
        assert ((prediction_truth is None) + (n_predict is None)) == 1
        if n_predict is not None:
            prediction_truth = [None for _ in range(n_predict)]

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
        positions = []  # true (during obs phase) and predicted positions
        force_fields = []

        if len(observed) == 2:
            positions = [observed[-1]]

        # encoder
        for obs1, obs2 in zip(observed[:-1], observed[1:]):
            ##LSTM Step
            hidden_cell_state, force_field = self.step(self.encoder, hidden_cell_state, obs1, obs2, goals, batch_split)
            force_fields.append(force_field)

        # initialize predictions with last position to form velocity. DEEP COPY !!!
        prediction_truth = copy.deepcopy(list(itertools.chain.from_iterable(
            (observed[-2:], prediction_truth)
        )))
        # decoder, predictions
        for obs1, obs2 in zip(prediction_truth[:-1], prediction_truth[1:]):
            if obs1 is None:
                obs1 = positions[-2].detach()  # DETACH!!!
            if obs2 is None:
                obs2 = positions[-1].detach()

            velocity = self.encoder_to_force_field(force_field, obs1, obs2, batch_split, observed=observed)
            # concat predictions
            # normals.append(normal)
            velocity = torch.stack(velocity, dim=0)
            positions.append(obs2.cpu() + velocity)

        # Pred_scene: Tensor [seq_length, num_tracks, 2]
        #    Absolute positions of all pedestrians
        # Rel_pred_scene: Tensor [seq_length, num_tracks, 5]
        #    Velocities of all pedestrians
        print(positions)
        pred_scene = torch.stack(positions, dim=0)
        rel_pred_scene = torch.stack(positions, dim=0)

        return rel_pred_scene, pred_scene
    
    def encoder_to_force_field(self, force_field, obs1, obs2, batch_split, observed):
        print("decoder")

        # Extract Force Field out of Encoder output
        obs1m = obs1.detach().cpu().numpy()
        obs2m = obs2.detach().cpu().numpy()
        velocities = obs2m - obs1m

        # Create a grid of points
        x_min = min(np.min(obs1m[:, 0]), np.min(obs2m[:, 0]))
        x_max = max(np.max(obs1m[:, 0]), np.max(obs2m[:, 0]))
        y_min = min(np.min(obs1m[:, 1]), np.min(obs2m[:, 1]))
        y_max = max(np.max(obs1m[:, 1]), np.max(obs2m[:, 1]))

        # Create a grid of points
        x = np.linspace(x_min, x_max, self.force_field_resolution)
        y = np.linspace(y_min, y_max, self.force_field_resolution)
        X, Y = np.meshgrid(x, y)

        forces = []

        # For each pedestrian
        for i in range(len(velocities)):
            # Get the velocity of the pedestrian
            vx, vy = velocities[i]

            # Scale the velocities to make them larger
            scale = 10
            vx *= scale
            vy *= scale

            # Create a vector field where each vector has the same direction as the pedestrian's velocity
            U = vx * np.ones_like(X)
            V = vy * np.ones_like(Y)

            # Manipulate the vector field based on the output of the encoder
            # print(len(force_field))
            encoder_output_reshaped = force_field[i].view(self.force_field_resolution, self.force_field_resolution, 2)
            # print(encoder_output_reshaped.shape)
            # print(encoder_output_reshaped)
            encoder_output = encoder_output_reshaped.detach().cpu().numpy()
            
            U += encoder_output[ : , : , 0]
            V += encoder_output[ : , : , 1]
            

            # Plot the vector field
            plt.figure()
            plt.quiver(X, Y, U, V)
            scale = 1
            # Plot the position of the pedestrian
            # plt.plot(obs2m[i, 0] / scale, obs2m[i, 1] / scale, 'ro')

            # Plot the trajectory of the pedestrian
            plt.plot([obs1m[i, 0] / scale, obs2m[i, 0] / scale], [obs1m[i, 1] / scale, obs2m[i, 1] / scale], 'r-', linewidth=2, color="red")
            if observed is not None:
                for j in range(len(velocities)):
                    past_positions = observed[:, j, :].detach().cpu().numpy()
                    color = "blue"
                    if j == i: color = "red"
                    for k in range(len(past_positions) - 1):
                        plt.plot([past_positions[k][0] / scale, past_positions[k+1][0] / scale], 
                                [past_positions[k][1] / scale, past_positions[k+1][1] / scale], 'b-', linewidth=2, color=color)


            plt.title(f'Vector field for pedestrian {i}')
            # Save the plot as an image file
            plt.savefig(f'vector_field_pedestrian_{i}.png')

            # Create a grid of points for interpolation
            points = np.column_stack([X.flatten(), Y.flatten()])

            # Create an array of vectors for interpolation
            vectors = np.column_stack([U.flatten(), V.flatten()])

            # Interpolate the force field to find the force at the pedestrian's current position
            current_position = obs2m[i]
            current_force = griddata(points, vectors, current_position, method='cubic')

            # Add the current force to the list of forces
            forces.append(torch.from_numpy(current_force))
        print(forces)
        return forces


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
        print("here")
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
