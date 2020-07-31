r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    reg = 0
    lr = 0.04
    wstd = 0.1
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0, 0, 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr_vanilla = 0.04
    lr_momentum = 0.004
    lr_rmsprop = 0.0004
    reg = 0.001
    # ========================
    return dict(wstd=wstd, lr_vanilla=lr_vanilla, lr_momentum=lr_momentum,
                lr_rmsprop=lr_rmsprop, reg=reg)


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr = 0.004
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**

1. Our initial expectations were to see that the lower the dropout would be, the accuracy will be higehr, while the loss 
   would be lower on the training set. That is because of overfitting, the model can learn very sophisticated relations 
   due to the many hidden layers. 
   if the training data would be limited, many of those insights would be lost, and it would be the result of sampling 
   noise, therefore it would exist in the training set, but not in the test set. This of course leads to overfitting 
   which is what we got in the training graph.
   Regarding the test graphs, as we can see, the low drop-out ( 0 ) actually performs better then the 0.4 and 0.8 dropout.
   This is against our expectations, we expected that the drop of 0.4 would be better then not using the dropout at all. 
   It might be becuase that our loss in unstable, although we are still getting acc above 80%.
2. We can see that the results using the low-dropout setting (0.4) are better then the results using 
   the high-dropout setting (0.8) both on the train and test.
   We think it happens due to underfitting- when the dropout set to 0.8 the inputs have high probability to dropped out
   so the model is not powerful enough and the model is over-regularized. We can assume from that, that the network
   did not learn enough. 

"""

part2_q2 = r"""
**Your answer:**

Yes it is possible. For example when the classification of one example changes to be true (and the others classification
stays the same) and in parallel some examples with very bad predictions keep getting worse.
An example, a dog image predicted at 0.7 to be a horse becomes predicted at 0.86 to be a horse).
With this phenomenon, it is possible for the test loss to increase while the test accuracy also increases.

"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**

We can see from the graph that L=2 gives us the best accuracy, while L=16 and L=8 give us the worst accuracy. 
We assume that L=16 is not trainable since the convolution netwerk involves to many pooling layers which downsample the features. If the features are downsampled too much we lose too much information,
thus the samples can't be classified properly. In addition, We assume the reason for that is connected to ReLU function. when dealing with ReLU there is a situation where a ReLU neuron is 
stuck in the negative side and always outputs 0, because the gradient of ReLU in the negative range is 0.
The risk for that to happen increases with the depth of the network as we can see in the graphs.

Attempts to address this issue: Lower learning rates often mitigates the problem. If not, use other functions then ReLU, and also using less pooling layers.

"""

part3_q2 = r"""
**Your answer:**

The two answers are a like with respect to the fact that the more layers you have the less accuracy you get (L=8). However, as opposed to the
former experiment we can observe that the larger the filters are in the L=8 graph the cnn learns and envolves into good results.
In general in that experiment, we can observe that the layer shape has its effect and produce better results. We can assume that the images comntain 
small detailes with the smaller layers smooth off while the bigger layers take that under consideration.

"""

part3_q3 = r"""
**Your answer:**

The results show us that L=1 and L=2 with the fixed K are getting the fastest best accuracy while the L=3 starts
slowly with smaller improvment. On the other hand, the L=4 doesn't improve at all. The more layers we have the less accuracy we get.
As we discussed above the change and modifing the hyperparameters would help improving the L=4 acuracy' however that would probably have an impact
over the former graphs. 

"""

part3_q4 = r"""
**Your answer:**

We have two sections: K=[32] fixed with L=8,16,32 and K=[64, 128, 256] fixed with L=2,4,8.
Both of the sections had a problem analyzing big numbers of filters combining with big L s. We suspect that it happens due to default initialization of the ReLU
function being done by PyTorch and the numerical errors we get when dealing with very small numbers.

In the first section, in comparing to 1.1 setion: we can see that L16 is getting much better results and L8 is improved.
In the second section, comparing to 1.3 secction: we can see that there is a significant improvement over the L4 while the L2 shows
minor improvement.

"""

part3_q5 = r"""
**Your answer:**

The model design is as follow:
((Conv -> Batch -> Dropout)*N  -> MaxPool)*N/P
  ^-------SKIP---------^
  While the last layer is only Conv layer instead of the "(Conv -> Batch -> Dropout)" without a SKIP.

  As we can see, comparing to the other sections, when enlarging the number of filters we have a problem with the results.
  In addition to that we can see that L6 and L2 have gained a high accuracy result pretty fast. The L2 shows the best result with
  high accuracy with the least amount of epochs.  

"""
# ==============
