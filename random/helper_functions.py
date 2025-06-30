" This document contains a variety of helper functions"

import matplotlib.pyplot as plt 


def plot_learned_trajectories(results, sample_idx=0, particle_idx=0):

    
    # Extract trajectory for the selected sample and particle (shape: [num_timesteps, 2])
    trajectory = results[sample_idx, :, particle_idx, :]  
    
    # Plot as (x, y) pairs
    plt.figure(figsize=(10, 6))
    plt.plot(trajectory[:, 0], trajectory[:, 1], 
             'b-', label=f'Particle {particle_idx}', linewidth=2)  # Solid blue line
    
    # Mark start (green) and end (red) points
    plt.scatter(trajectory[0, 0], trajectory[0, 1], c='g', s=100, label='Start', zorder=3)
    plt.scatter(trajectory[-1, 0], trajectory[-1, 1], c='r', s=100, label='End', zorder=3)
    
    plt.title(f'2D Trajectory for Sample {sample_idx}, Particle {particle_idx}')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')  # Ensures equal scaling for x and y axes
    plt.show()

def plot_multiple_particles(results, true, buff, start, sample_idx=0, particle_indices=[0, 1, 2, 3]):

    print(len(true))
    print(len(results))
    plt.figure(figsize=(10, 6))
    for r in range(len(results)):
        for p in particle_indices:
            traj = results[r][sample_idx, :, p, :].cpu()
            plt.plot(traj[:, 0], traj[:, 1], label=f'Particle {p}')
            plt.scatter(traj[0, 0], traj[0, 1], c='g', s=50)
            plt.scatter(traj[-1, 0], traj[-1, 1], c='r', s=50)

    for t in range(buff * len(results)): 
        print('total range')
        print(buff * len(results))
        for p in particle_indices: 
            true_traj = true[5][start + t, p, :]
            plt.scatter(true_traj[0], true_traj[1], c='b', s=50)
    plt.xlim(0, 5)
    plt.ylim(-1, 5)
    plt.title(f'Trajectories for Sample {sample_idx}')
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('full_run.png')
