# Code for Bayesian model of the co-evolution of language and mindreading
In the folder [python_code](https://github.com/marieke-woensdregt/model_coevolution_language_mindreading/blob/master/python_code) in this repository you can find all the Python code I used to implement my Bayesian model of the co-evolution of language and mindreading (including development, iterated learning and biological evolution). I divided the code up into several modules which implement different parts of the model (e.g. agents, languages, contexts and so on). Then there are four main modules which import all these modules in order to run a full simulation. The module 'run_learner_speaker.py' runs a simulation of a single learner learning from a single speaker (as described in Woensdregt et al., 2016), and 'run_learner_pop.py' does the same for a learner receiving data from a population of different speakers. The module 'run_pop_iteration.py' runs a simulation of a whole population transmitting languages over generations using iterated learning, in which different selection pressures can be switched on or off. And finally, the module 'run_evolvability_analysis.py' loads in the data of a population of literal agents which has already run to convergence, and inserts a pragmatic 'mutant' agent into this population, followed by a simulation of a combined cultural+biological evolution model (where the gene for being 'pragmatic' or 'literal' is transmitted genetically). Each of these main modules starts with a list of parameter settings right after the import statements.
There is also a module named 'plots.py' which contains many different plotting functions. 

Throughout these modules I used a mixture of object-oriented programming and regular functions, and explained how each function, class or method works using docstrings. I also used long and intelligible variable and function names, which should hopefully make the code relatively easy to read.
**Note** that some of these simulations take quite a long time to run (learners require a relatively large amount of observations because they are subjected to a joint inference task), so it might be worth outsourcing the running of large simulations to a computer cluster or similar.





### References
Woensdregt, M., Kirby, S., Cummins, C. & Smith, K. (2016). Modelling the co-development of word learning and perspective-taking. Proceedings of 38th Annual Meeting of the Cognitive Science Society.
