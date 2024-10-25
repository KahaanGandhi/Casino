import torch
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

class GravityInteraction(MessagePassing):
    def __init__(self):
        super(GravityInteraction, self).__init__(aggr='add')  # Summing forces from all neighbors

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j, epsilon=1e-6, G=1):
        pos_i = x_i[:, :2]
        pos_j = x_j[:, :2]
        mass_i = x_i[:, 4:5]
        mass_j = x_j[:, 4:5]

        # Calculate gravitational force vector
        distance = torch.norm(pos_i - pos_j, dim=1).unsqueeze(1)
        force_mag = G * mass_i * mass_j / (distance ** 2 + epsilon)
        force_direction = (pos_j - pos_i) / (distance + epsilon)
        force = force_mag * force_direction
        return force

    def update(self, aggr_out, x):
        velocity = x[:, 2:4]
        mass = x[:, 4:5]
        acceleration = aggr_out / mass

        # Update positions and velocities
        new_velocity = velocity + acceleration * 0.05  # Time step of 0.05
        new_positions = torch.cat((x[:, :2] + new_velocity, new_velocity, mass), dim=1)
        return new_positions

def simulate(data, model, timesteps=1000):
    positions_history = [data.x[:, :2].detach().numpy()]

    for t in range(timesteps):
        # Compute the forces and update node features (positions and velocities)
        updated_features = model(data.x, data.edge_index)
        data.x = updated_features  # Update graph node features
        positions_history.append(data.x[:, :2].detach().numpy())

    return np.array(positions_history)

# Function to generate initial conditions
def generate_initial_conditions(num_particles):
    torch.manual_seed(777)
    
    base_positions = torch.tensor([[40, 40], [40, 60], [55, 75]], dtype=torch.float32)
    positions = base_positions[torch.randint(0, base_positions.shape[0], (num_particles,))] + (torch.rand(num_particles, 2) - 0.5) * 10
    base_velocities = torch.tensor([[-0.2, 1.5], [0.1, -1.5], [1.5, -0.5]], dtype=torch.float32) * 0.1
    velocities = base_velocities[torch.randint(0, base_velocities.shape[0], (num_particles,))] + (torch.rand(num_particles, 2) - 0.5) * 0.5
    masses = 20 + (torch.rand(num_particles, 1) * 10)
    
    return positions, velocities, masses

# Update positions, trais and limits
def update(frame):
    scat.set_offsets(positions_history[frame])

    for i, trail in enumerate(trails):
        trail.set_data(positions_history[:frame, i, 0], positions_history[:frame, i, 1])

    current_positions = positions_history[frame]
    x_min, x_max = current_positions[:, 0].min(), current_positions[:, 0].max()
    y_min, y_max = current_positions[:, 1].min(), current_positions[:, 1].max()
    # ax.set_xlim(x_min - 5, x_max + 5)
    # ax.set_ylim(y_min - 5, y_max + 5)
    ax.set_xlim(30, 90)
    ax.set_ylim(30, 90)

    return scat, *trails


num_particles = 4
positions, velocities, masses = generate_initial_conditions(num_particles)

# Create fully connected graph
node_features = torch.cat([positions, velocities, masses], dim=1)
edge_index = torch.tensor([[i, j] for i in range(num_particles) for j in range(num_particles) if i != j], dtype=torch.long).t()
data = Data(x=node_features, edge_index=edge_index)

# Instantiate model and run simulation
model = GravityInteraction()
positions_history = simulate(data, model, timesteps=1000)

# Animate results
positions_history_np = np.array(positions_history)
x_min, x_max = positions_history_np[:, :, 0].min(), positions_history_np[:, :, 0].max()
y_min, y_max = positions_history_np[:, :, 1].min(), positions_history_np[:, :, 1].max()
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_facecolor('black')
colors = plt.cm.viridis(np.linspace(0, 1, num_particles))
scat = ax.scatter(positions_history[0][:, 0], positions_history[0][:, 1], s=100, c=colors, alpha=0.8)
ax.set_xlim(x_min - 5, x_max + 5)
ax.set_ylim(y_min - 5, y_max + 5)
trails = [ax.plot([], [], '-', lw=2, alpha=0.5, color=colors[i])[0] for i in range(num_particles)]
ani = FuncAnimation(fig, update, frames=len(positions_history), interval=50, blit=True)
plt.style.use('dark_background')
ax.set_title('Chaotic Particle Orbits', color='white', fontsize=16)
ax.set_xlabel('X Position', color='white', fontsize=12)
ax.set_ylabel('Y Position', color='white', fontsize=12)
plt.show()