
# Differentiable Neural Memory Computer

- DNMC: This is an implementation of a Feed Forward Network which uses another Feed Forward Network as its external Memory

- DNMC^2: This is an implementation of a Feed Forward Network which uses a DNMC as its external Cognitive Memory.

- DNMC^2_remember_memory: This is an DNMC^2 implementation wich remembers the state of the Cognitive Memory between two time_steps.

- DNMC^2_2.0: This is an DNMC^2 which remembers the state of its Cognitive Memory and the Controller gets as input the hall state of the Cognitive Memory (not only the weights and outputs of the MemoryController also from the MemoryMemoryController).

- DNMC^2_3.0: This is an DNMC^2_2.0 with a "more extern" memory that tries to make sure that the hall construction is not only a massive recurrent cell effect.

DNMC^2 has a kind of cognition a little bit similar to https://arxiv.org/pdf/1703.04361.pdf%3E through reading several times from the memory while the ith read influences the i+1th read.
It is like you read a pointer in the first read to access the array in the next read, only that everything is differentiable.

The .train() methode generates random sequences between 0 and 1, and trains the Network for the copy function.

Please feel free to write more train()  methodes to test especially DNMC^2 on more complicated problems, i dont have the calculation power at the moment.

It learns the Copy Function through time and some similar problems.

I dont know if it is turing complete, if someone has an idea how to proof it please write me.





You can also use an LSTM Network but its better to use  a FeedForwardController so that you can be sure that it is
the external Memory which is working.

I think its  fascinating that the hidden layers of the controller can be smaller than the input_size and output_size and it still
can manage to learn the copy function.

My mathematical idea was to programm something you could call a functional analytic neural Object, because the Controller gets the weights of the second Network as part of its input.
So it is some kind of combination of computational theory, analysis and functionalanalysis.

More about a function as an object see "Axiomatic Foundations of Mathematics" from John von Neumann.
