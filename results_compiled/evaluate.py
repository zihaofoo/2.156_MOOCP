import numpy as np
import scipy
import matplotlib.pyplot as plt
from linkage_utils import evaluate_submission, draw_mechanism, solve_mechanism, to_final_representation, evaluate_mechanism, is_pareto_efficient
from pymoo.indicators.hv import HV

# evaluate_submission()


# this function does a basic check that the linkage works
def check_grashof_condition(a, b, c, d):
    # Calculate the sums of the shortest and longest links
    min_sum = min(a, b, c, d) + max(a, b, c, d)
    other_sum = sum([a, b, c, d]) - min(a, b, c, d) - max(a, b, c, d)
    if min_sum < other_sum:
        return True
    else:
        return False

#this function moves the node a little to try to satisfy the grashof condition. It moves a node on the longer line 10% closer by default
#returns the adjusted
def correct_grashof_violation(x0, Link, move_percentage=0.1):
    
    for i in range(100):
    # Calculate the lengths of the linkages
        lengths = [np.linalg.norm(np.array(x0[p1]) - np.array(x0[p2])) for p1, p2 in Link]
        
        if check_grashof_condition(*lengths):
            return x0
        else:    
            # Find the index of the longest linkage
            print("shortening")
            longest_link_index = lengths.index(max(lengths))
            
            # Find the nodes connected by the longest link
            node1, node2 = Link[longest_link_index]
            
            # Calculate the vector from node1 to node2
            vector_to_node2 = [x0[node2][0] - x0[node1][0], x0[node2][1] - x0[node1][1]]
            
            # Calculate the displacement vector for node2
            displacement_vector = [move_percentage * vector_to_node2[0], move_percentage * vector_to_node2[1]]
            
            # Update the position of node2
            x0[node2] = [x0[node2][0] - displacement_vector[0], x0[node2][1] - displacement_vector[1]]
    
    print("Failed to satisfy Grashof Condition")
    return x0

class Mechanism:
    """Object class for the mechanism. Returns a Grashof mechanism with n nodes. Inputs: n (int) - number of nodes"""
    def __init__(self, n:int, C:object, x0:object, fixed_nodes:object, motor:object, links:object):
        self.n = n
        self.C = C
        self.x0 = x0
        self.fixed_nodes = fixed_nodes
        self.motor = motor
        self.links = links
    def get_C(self):
        return self.C
    def set_C(self, C):
        self.C = C
    def get_x0(self):
        return self.x0
    def set_x0(self, x0):
        self.x0 = x0
    def get_fixed_nodes(self):
        return self.fixed_nodes 
    def set_fixed_nodes(self, fixed_nodes):
        self.fixed_nodes = fixed_nodes    
    def get_motor(self):
        return self.motor 
    def set_motor(self, motor):
        self.motor = motor   
    def get_links(self):
        return self.links 
    def set_links(self, links):
        self.links = links   

def generate_grashof_mechanism():
    """Generate random link lengths within a unit square"""
    C = np.array([[0,1,1,0],\
              [1,0,0,1],\
              [1,0,0,1],\
              [0,1,1,0]], dtype = float)
    fixed_nodes = np.array([0,1])
    motor = np.array([0,2])

    a = np.random.uniform(0.1, 0.5)             # Length of the input link (within [0.1, 0.5])
    b = np.random.uniform(a + 0.1, 1.0 - 0.1)   # Length of the coupler link (within [a + 0.1, 0.9])
    c = np.random.uniform(0.1, 0.5)             # Length of the output link (within [0.1, 0.5])
    d = np.random.uniform(c + 0.1, 1.0 - a)     # Length of the fixed link (within [c + 0.1, 1.0 - a])

    # Calculate the angles for the four-bar Grashof mechanism
    theta2 = np.random.uniform(0.0, 2*np.pi)    # Angle of the coupler link (theta2)
    theta3 = np.random.uniform(0.0, 2*np.pi)    # Angle of the output link (theta3)

    # Check for Grashof condition
    if check_grashof_condition(a, b, c, d):
        # print(a,b,c,d)
        x_init, y_init = 0.0, 0.0
        x0 = np.array([[x_init, y_init], [x_init + d, y_init], [x_init + a * np.cos(theta2), y_init + a * np.sin(theta2)], \
                       [x_init + c * np.cos(theta3), y_init + c * np.sin(theta3)]], dtype = float)
        x_min, y_min = np.min(x0[:,0]), np.min(x0[:,1])
        x0 = x0 - np.array([[x_min, y_min]], dtype = float)
        valid, _, _, _ = solve_mechanism(C, x0, fixed_nodes, motor, device = "cpu", timesteps = 2000)
        if valid:    
            return x0
        else: 
            return generate_grashof_mechanism()
    else:
        return generate_grashof_mechanism()

def generate_5bar_mechanism():
    """Returns a valid 5-bar mechanism."""
    C = np.array([[0,1,1,0,0],\
                  [1,0,0,1,0],\
                  [1,0,0,1,1],\
                  [0,1,1,0,1],\
                  [0,0,1,1,0]], dtype = float)
    x0 = generate_grashof_mechanism()       # x0 of 4-bar linkage
    x0 = np.vstack((x0, np.array([np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0)])))
    fixed_nodes = np.array([0,1])
    motor = np.array([0,2])

    links = np.array([[0,1],\
                     [0,2],\
                     [1,3],\
                     [2,3],\
                     [2,4],\
                     [3,4]], dtype = int)
    mech_5 = Mechanism(5, C, x0, fixed_nodes, motor, links)
    # draw_mechanism(mech_5.C, mech_5.x0, mech_5.fixed_nodes, mech_5.motor)
    return mech_5

def generate_valid_mechanism(n_nodes:int):
    mech = generate_5bar_mechanism()
    n_add = n_nodes - 5             # Number of nodes to be added

    if n_add < 0:
        raise ValueError('Number of nodes is less than 5. Invalid mechanism.')
    
    for i1 in range(n_add):
        coin_toss = scipy.stats.bernoulli.rvs(p = 0.5)      # Coin toss to decide d_decision or t_decision
        if coin_toss == 0:
            mech = d_decision(mech)
        else:
            mech = t_decision(mech)
        
    return mech

def d_decision(mech:object, hyp:float = 0.5):
    """Function that returns a connectivity matrix and node position of a n+1 linkage. \n Inputs: mech (mechanism object)"""
    C = mech.get_C()
    n = np.shape(C)[0]      # Number of nodes
    x0 = mech.get_x0()
    motor = mech.get_motor()
    fixed_nodes = mech.get_fixed_nodes()
    links = mech.get_links()
    num_links = np.shape(links)[0] - 3
    new_index = np.random.randint(low = 1, high = num_links + 1)
    linkage_nodes = x0[links[new_index + 2],:]              # Coordinates of two nodes in the selected link
    midpoint_link = np.mean(linkage_nodes, axis = 0)
    new_points = midpoint_link + np.array([[np.random.uniform(-hyp, hyp), np.random.uniform(-hyp, hyp)]], dtype = float)        # Coordinate of new node
    C = np.hstack((np.vstack((C, np.zeros((1,n)))), np.zeros((n+1,1))))
    
    # print('Linkage Nodes', linkage_nodes)
    # print('Midpoint', midpoint_link)
    # print('New points', new_points)

    for i1 in range(2):    
        C[links[new_index + 2][i1], n] = 1
        C[n, links[new_index + 2][i1]] = 1

    x0 = np.vstack((x0, new_points))
    links = np.vstack((links, np.array([[links[new_index + 2][0], n], [links[new_index + 2][1], n]])))
    mech.x0, mech.C, mech.links = x0, C, links
    return mech

def t_decision(mech:object, hyp:float = 0.5):
    """Function that returns a connectivity matrix and node position of a n+1 linkage. \n Inputs: mech (mechanism object)"""
    C = mech.get_C()
    n = np.shape(C)[0]      # Number of nodes
    x0 = mech.get_x0()
    motor = mech.get_motor()
    fixed_nodes = mech.get_fixed_nodes()
    links = mech.get_links()
    num_links = np.shape(links)[0] - 3
    new_index = np.random.randint(low = 2, high = num_links + 1)
    linkage_nodes = x0[links[new_index + 2],:]              # Coordinates of two nodes in the selected link
    midpoint_link = np.mean(linkage_nodes, axis = 0)
    new_points = midpoint_link + np.array([[np.random.uniform(-hyp, hyp), np.random.uniform(-hyp, hyp)]], dtype = float)        # Coordinate of new node
    C = np.hstack((np.vstack((C, np.zeros((1,n)))), np.zeros((n+1,1))))
    
    for i1 in range(2):    
        C[links[new_index + 2][i1], n] = 1
        C[n, links[new_index + 2][i1]] = 1
    
    index_del = np.hstack((np.array([0,1]), np.array(links[new_index + 2])))
    n_list = np.delete(np.arange(n), index_del)
    target_n = np.random.choice(n_list, size = 1)
    C[target_n, n] = 1
    C[n, target_n] = 1
    C[links[new_index + 2][0], links[new_index + 2][1]] = 0
    C[links[new_index + 2][1], links[new_index + 2][0]] = 0

    x0 = np.vstack((x0, new_points))
    links = np.vstack((links, np.array([[links[new_index + 2][0], n], [links[new_index + 2][1], n]])))
    links = np.vstack((links, np.array([target_n[0], n])))
    links = np.delete(links, new_index + 2, axis = 0)

    valid, _, _, _ = solve_mechanism(C, x0, fixed_nodes, motor, device = "cpu", timesteps = 2000)
    if valid:
        """Accept solution if mechanism is valid."""
        mech.x0, mech.C, mech.links = x0, C, links
        return mech
    else:
        """Reject solution and reattempt randomizer."""
        return t_decision(mech)


def plot_HV(F, ref):

    #Plot the designs
    plt.scatter(F[:,1],F[:,0])

    #plot the reference point
    plt.scatter(ref[1],ref[0],color="red")

    #plot labels
    plt.xlabel('Material Use')
    plt.ylabel('Chamfer Distance')

    #sort designs and append reference point
    sorted_performance = F[np.argsort(F[:,1])]
    sorted_performance = np.concatenate([sorted_performance,[ref]])

    #create "ghost points" for inner corners
    inner_corners = np.stack([sorted_performance[:,0], np.roll(sorted_performance[:,1], -1)]).T

    #Interleave designs and ghost points
    final = np.empty((sorted_performance.shape[0]*2, 2))
    final[::2,:] = sorted_performance
    final[1::2,:] = inner_corners

    #Create filled polygon
    plt.fill(final[:,1],final[:,0],color="#008cff",alpha=0.2)

    #Specify reference point
    ref_point = np.array([0.1, 10])

    #Calculate Hypervolume
    ind = HV(ref_point)
    hypervolume = ind(F)

    #Print and plot
    print('Hyper Volume ~ %f' %(hypervolume))
    plot_HV(F, ref_point)

# Initialize an empty list to store target curves
target_curves = []

# Loop to read 6 CSV files and store data in target_curves list
for i in range(6):
    # Load data from each CSV file and append it to the list
    target_curves.append(np.loadtxt('./data/%i.csv'%(i),delimiter=','))

target_index = 1
target_curve = np.array(target_curves[target_index])

# to_final_representation(C,x0,fixed_nodes,motor,target)
# ref_point = np.array([0.1, 10])
# ind = HV(ref_point)
# hypervolume = ind(results.F)
# print('Hyper Volume ~ %f' %(hypervolume))
# plot_HV(results.F, ref_point)

n_nodes = np.arange(start=5, step=1, stop=6)
hyp_vol = np.zeros(np.shape(n_nodes))
num_MC = np.int_(1E6)
mechanisms = []
ref_point = np.array([10, 0.1])

fig1, ax1 = plt.subplots(figsize = (10,10))
cost_mat = np.zeros((num_MC, 2), dtype = float)
for i2 in range(len(n_nodes)):
    for i1 in range(num_MC):
        mech = generate_valid_mechanism(n_nodes[i2])
        C, x0, fixed_nodes, motor = mech.C, mech.x0, mech.fixed_nodes, mech.motor
        mechanisms.append(to_final_representation(C, x0, fixed_nodes, motor, n_nodes[i2] - 1))
        valid, CD, material, sol = evaluate_mechanism(C, x0, fixed_nodes, motor, target_curve, idx=None,device='cpu',timesteps=2000)
        cost_mat[i1,:] = [material, CD]

        if i1 % 1000 == 0:
            print('n =', n_nodes[i2], 'iteration: ', i1)
            # int_cost = cost_mat[:num_MC,:]
            # pareto_mask = is_pareto_efficient(int_cost)
            # pareto_front = int_cost[pareto_mask]
            # ind = HV(ref_point)
            # hypervolume = ind(pareto_front)
            # print('Hyper Volume ~ %f' %(hypervolume))


    pareto_mask = is_pareto_efficient(cost_mat)
    pareto_front = cost_mat[pareto_mask]

    # print('Pareto Front', pareto_front)
    ref_point = np.array([10, 0.1])
    # hypervolume= calculate_hypervolume(pareto_front, ref_point)

    ind = HV(ref_point)
    hypervolume = ind(pareto_front)
    np.savetxt("Monte_Carlo.csv", np.array(mechanisms), delimiter=",")

    print('Hyper Volume ~ %f' %(hypervolume))
    hyp_vol[i2] = hypervolume
    ax1.scatter(pareto_front[:,0], pareto_front[:,1])
    ax1.set_ybound(lower = 0, upper = 0.1)
    ax1.set_xbound(lower = 0, upper = 10)

ax1.plot(n_nodes, hyp_vol)
plt.show()



