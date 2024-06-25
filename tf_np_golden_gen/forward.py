import numpy as np


def layernorm(x, gamma, beta, epsilon=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    x_normalized = (x - mean) / np.sqrt(variance + epsilon)
    y = gamma * x_normalized + beta
    
    return y

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))


B = 1  # batch size
T = 16 # sequential length
C = 48 # embedding dimensionality
n_head = 6
head_dim = C // n_head  # dimension of each head

gamma = np.ones((1, 1, C))
beta = np.zeros((1, 1, C))

x = np.random.rand(B, T, C) # input
attn_input_x = x 
x = layernorm(x,gamma,beta)

#attention
# This is for hardware, in hardware, we do operation for each head
# So I split weights into heads
Q_weights = np.random.randn(n_head, C, head_dim)  
K_weights = np.random.randn(n_head, C, head_dim)
V_weights = np.random.randn(n_head, C, head_dim)
Q_bias = np.random.randn(n_head, head_dim)
K_bias = np.random.randn(n_head, head_dim)
V_bias = np.random.randn(n_head, head_dim)


q_heads = np.zeros((n_head, B, T, head_dim))
k_heads = np.zeros((n_head, B, T, head_dim))
v_heads = np.zeros((n_head, B, T, head_dim))
p_heads = np.zeros((n_head, B, T, T)) # p = q*k^t
y_heads = np.zeros((n_head, B, T, head_dim)) # y=p_softmax*v

for i in range(n_head):
    q = np.matmul(x, Q_weights[i]) + Q_bias[i]
    # print(q.shape) #(B, T, head_dim)
    q_heads[i] = q
    
    k = np.matmul(x, K_weights[i]) + K_bias[i]
    k_heads[i] = k
    
    v = np.matmul(x, V_weights[i]) + V_bias[i]
    v_heads[i] = v

    p = np.matmul(q, k.transpose((0, 2, 1)))  # (B, T, T)
    p_heads[i] = p
    
    p_softmax = softmax(p, axis=-1)  # (B, T, T)
    
    y = np.matmul(p_softmax, v)  # (B, T, head_dim)
    y_heads[i] = y

y_concat = np.concatenate(y_heads, axis=-1) 
# print(y_concat.shape) (B,T,C)
project_weights = np.random.randn(C, C)
project_bias = np.random.randn(C)

y_projected = np.matmul(y_concat, project_weights) + project_bias
# print(y_projected.shape) #(B,T,C)

#finish of attention

y = y_projected + attn_input_x
MLP_input_x = y
y = layernorm(y,gamma, beta)

#MLP

W_fc = np.random.randn(C, 4 * C)
b_fc = np.random.randn(4 * C)
W_proj = np.random.randn(4 * C, C)
b_proj = np.random.randn(C)

x_fc = np.matmul(y, W_fc) + b_fc
x_gelu = gelu(x_fc)
x_proj = np.matmul(x_gelu, W_proj) + b_proj
# print(x_proj.shape)
# finish of MLP

y = x_proj + MLP_input_x

# Finish of decoder