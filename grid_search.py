def grid_search(opt_type):
    learning_rate_samples = [6e-2, 6e-3, 6e-4]
    weight_decay_samples = [1e-1, 0]
    beta1 = [0.9]
    beta2 = [0.95]
    if opt_type == 'adam' or opt_type == 'adamw':
        beta1 = [0.8, 0.9]
        beta2 = [0.9, 0.95]
    combinations = []
    for l in learning_rate_samples:
        for w in weight_decay_samples:
            for b1 in beta1:
                for b2 in beta2:
                    combinations.append((l, w, b1, b2))
    return combinations
