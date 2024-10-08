import pickle
import os

def safe_pickle(data, directory_path, file_name):
    # Check if the directory exists, and if not, create it
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    
    # Construct the full path for the file
    file_path = os.path.join(directory_path, file_name)
    
    # Open the file in binary-write mode and pickle the data to it
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)


def plot_actions(ax, action, t_c, dt):
    if action.ndim == 1:
        action.reshape((-1, 1))
    for i in range(action.shape[1]):
        if i == 0:
            action_c = action[(t_c//dt).astype(int), i]
            ax.plot(t_c, action_c, f'C{i}')
        else:
            axtwin = ax.twinx()
            action_c = action[(t_c//dt).astype(int), i]
            axtwin.plot(t_c, action_c, f'C{i}')