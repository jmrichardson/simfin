from hyperopt.pyll.stochastic import sample
space = {
    'x': hp.loguniform('random_strength', np.log(0.0001), np.log(5)),
    'random_strength': round(sample(hp.loguniform('learning_rate', np.log(1), np.log(20)))),
}

print(sample(space))