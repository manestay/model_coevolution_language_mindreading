# Code for Bayesian model of the co-evolution of language and mindreading

## General
In the folder [python_code](https://github.com/marieke-woensdregt/model_coevolution_language_mindreading/tree/master/python_code) in this repository you can find all the Python code I used to implement my Bayesian model of the co-evolution of language and mindreading (including development, iterated learning and biological evolution). I subdivided the code into a folder containing the core modules, a folder containing the main scripts for running a full simulation, and a folder with scripts for analysis and plotting. 
In the folder [code_for_running_on_cluster](https://github.com/marieke-woensdregt/model_coevolution_language_mindreading/tree/master/code_for_running_on_cluster) you can find the code you need to run a batch of simulations as an array job on a Grid Engine computer cluster (works on the [Open Grid Scheduler](http://gridscheduler.sourceforge.net/) batch system on Scientific Linux 7 at least). The example allows you to create an array job on a cluster which runs the same simulation (2 independent runs of a single learner learning from a single speaker for 30 observations) but looping through each of the possible speaker lexicons (for a lexicon with 3 meanings and 3 signals in this case; 343 possible lexicons in total). 


### References
Woensdregt, M., Kirby, S., Cummins, C. & Smith, K. (2016). [Modelling the co-development of word learning and perspective-taking.](https://mindmodeling.org/cogsci2016/papers/0222/paper0222.pdf) Proceedings of 38th Annual Meeting of the Cognitive Science Society.
