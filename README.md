
#Differentiable Neural Memory Computer

#DNMC:This is an implementation of a Feed Forward Network which uses another Feed Forward Network as its external Memory

#DNMC^2:This is an implementation of a Feed Forward Network which uses a DNMC as its external Memory.

DNMC^2 has a kind of cognition a little bit similar to https://arxiv.org/pdf/1703.04361.pdf%3E: through reading several times from the memory while the ith read influences the i+1th read.
It is like you read a pointer in the first read to access the array in the next read, only that everything is differentiable.

The .train() methode generates random sequences between 0 and 1 , and trains the Network for the copy function.

Please feel free to write more train()  methodes to test especially DNMC^2 on more complicated problems, i dont have the calculation power at the moment.

It learns the Copy Function through time and some similar problems in less than 5000 iterations.

I dont know if it is turing complete, if someone has an idea how to proof it please write me.





You can also use an LSTM Network but its better to use  a FeedForwardController so that you can be sure that it is 
the external Memory which is working.

I think its  fascinating that the hidden layers of the controller can be smaller than the input_size and output_size and it still 
can manage to learn the copy function.

My mathematical idea was to programm something you could call a functional analytic neural Object, because the Controller gets the weights of the second Network as part of its input.
So it is some kind of combination of computational theory, analysis and functionalanalysis.

More about a function as an object see "Axiomatic Foundations of Mathematics" from John von Neumann. <3


