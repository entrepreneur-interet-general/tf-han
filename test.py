import text_classifier as tcl

exp_dir = "/Users/victor/Documents/Tracfin/dev/han/experiments"

exp = tcl.Experiment(exp_dir)

exp.run(n_runs=3, verbose=1)
