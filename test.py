import text_classifier as tcl

conf_path = "/Users/victor/Documents/Tracfin/dev/han/default_conf.yml"

exp = tcl.Experiment(conf_path=conf_path)

exp.run(n_runs=3, verbose=1)
