
#Differentiable Neural Memory Computer

Dont know if it is worth this name because i dont know if it is turing complete.

But it learns the copy function through time in less than 5000 iteration steps.




#DNMC

This is an implementation of a Feed Forward Network which uses another Feed Foward Network as an external Memory




#DNMC^2

This is an implementation of a Feed Forward Network which uses a DMNC as its external Memory






The .train() methode generates random sequences between 0 and 1 , and trains the Network for the copy function.

You can also use an LSTM Network but its better to use  a FeedForwardController so that you can be sure that it is 
the external Memory which is working.

I think its  fascinating that the hidden layers of the controller can be smaller than the input_size and output_size and it still 
can manage to learn the copy function.

Someone who has a mathematical idea of why this computation Graph is working please write me. 

Please feel free to write more train()  methodes to test especially DNMC^2 on more complicated problems, i dont have the calculation power at the moment.
