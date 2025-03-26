from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

grid_size = 5  # 預設網格大小
grid = np.zeros((grid_size, grid_size))
start_pos = None
goal_pos = []
dead_pos = []
policy = {}  # 全局策略變數

actions = {'↑': (-1, 0), '↓': (1, 0), '←': (0, -1), '→': (0, 1)}

def get_valid_actions(x, y):
    valid_actions = {}
    for action, (dx, dy) in actions.items():
        nx, ny = x + dx, y + dy
        if 0 <= nx < grid_size and 0 <= ny < grid_size:
            valid_actions[action] = (nx, ny)
    return valid_actions

@app.route('/')
def index():
    return render_template('index.html', grid_size=grid_size)

@app.route('/set_grid', methods=['POST'])
def set_grid():
    global grid_size, grid, start_pos, goal_pos, dead_pos, policy
    data = request.json
    grid_size = int(data['size'])
    grid = np.zeros((grid_size, grid_size))
    start_pos = None
    goal_pos = []
    dead_pos = []
    policy = {}
    return jsonify({'message': 'Grid size updated', 'grid_size': grid_size})

@app.route('/set_cell', methods=['POST'])
def set_cell():
    global start_pos, goal_pos, dead_pos
    data = request.json
    x, y, cell_type = data['x'], data['y'], data['type']
    
    if cell_type == 'start':
        start_pos = (x, y)
    elif cell_type == 'goal':
        goal_pos.append((x, y))
    elif cell_type == 'dead':
        dead_pos.append((x, y))
    
    return jsonify({'message': 'Cell updated'})

@app.route('/get_policy', methods=['GET'])
def get_policy():
    global policy
    random_actions = list(actions.keys())
    policy = {f"{i},{j}": np.random.choice(random_actions)
              for i in range(grid_size) for j in range(grid_size)
              if (i, j) not in goal_pos and (i, j) not in dead_pos}
    return jsonify({'policy': policy})

@app.route('/value_iteration', methods=['POST'])
def value_iteration():
    global policy
    gamma = 0.9  # 折扣因子
    theta = 1e-4  # 迭代停止閥值
    V = np.zeros((grid_size, grid_size))
    
    while True:
        delta = 0
        new_V = np.copy(V)
        
        for i in range(grid_size):
            for j in range(grid_size):
                if (i, j) in goal_pos:
                    new_V[i, j] = 1
                    continue
                if (i, j) in dead_pos:
                    new_V[i, j] = -1
                    continue
                
                max_value = float('-inf')
                best_action = None
                for action, (nx, ny) in get_valid_actions(i, j).items():
                    reward = -0.04  # 移動懲罰
                    if (nx, ny) in goal_pos:
                        reward = 1
                    elif (nx, ny) in dead_pos:
                        reward = -1
                    
                    value = reward + gamma * V[nx, ny]
                    if value > max_value:
                        max_value = value
                        best_action = action
                
                new_V[i, j] = max_value
                policy[f"{i},{j}"] = best_action
                delta = max(delta, abs(V[i, j] - new_V[i, j]))
        
        V = new_V
        if delta < theta:
            break
    
    return jsonify({'policy': policy, 'values': V.tolist()})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
