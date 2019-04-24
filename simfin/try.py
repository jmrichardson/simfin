from hyperopt.pyll.stochastic import sample
space = {
    'x': hp.loguniform('random_strength', np.log(0.0001), np.log(5)),
    'y': hp.uniform('depth', 6, 10),
}

print(sample(space))