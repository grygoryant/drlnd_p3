[image1]: img/nets.png "Actor and Critic nets"
[image2]: img/scores_sc.png "Average rewards with shared critic"
[image3]: img/scores_no_sc.png "Average rewards w/o shared critic"
[image4]: img/run_no_sc.gif "Trained w/o shared critic"
[image5]: img/run_sc.gif "Trained with shared critic"

# Report

## Learning algorithm

In order to solve this environment I've extended my DDPG implementation from the previous project. I've used two different neural network architecures: one for the Actor, and one for the Critic. It also uses a shared replay buffer and soft target networks updates. 

![Actor and Critic nets][image1]

First of all, shared replay buffer has been implemented: each agent's environment observation is buffered into common replay buffer. After sufficient amount of samples are available in buffer, each agent is able to sample environment states randomly (with the batch size of 256) from that buffer to learn. Each agent learns every time step and gets 1 update with a factor of 0.001 for their target nets. Adam optimizer has been used for both Actor and Critic nets with learning rates 0.0001. I've used the discount factor 0.99 for training. Critic network also has dropouts with drop_prob of 0.2.

It's also necessary no mention that Ornstein–Uhlenbeck process has been used to add noise to resulting actions during learning to make agent explore the environment. The important thing for convergence here is to slowly decrease the noise during learning. Before I've figured that out my training process was really long and in most cases it didn't converge with average score drops after approximately half of trainig.

## Rewards

Working on this project I've been experimenting with algorithms, hyperparameters and network architectures for quite a lot of time. I would especially like to highlight two experiments (both were able to solve the environment but with different agents behavior). Also note that I didn't break training process after reaching target avarage score of 0.5 to figure out if it's possible to improve agents' behavior even more.

### Shared buffer, no shared critic

For this experiment each agent had it's own isolated critic, learning only with it's actor. This algorithm solved the environment in 656 episodes with the following average rewards: 

![Average rewards][image3]

When I was checking trained agents, I've noticed their strange behavior, which made me thinking about finding the solution for avoiding such excess movements:

![Trained w/o shared critic][image4]

Then I've came up with the second experiment, where I've used a shared critic to cut off all excess moves.

### Shared buffer, shared critic

This approach allowed agents to learn just in 497 episodes, giving the following rewards chart:

![Average rewards][image2]

Trained agents behave just like I wanted:

![Trained with shared critic][image5]

## Future work

In future I'd like to try to implement the real MADDPG with critics taking all available states and actions. In addition, I guess it's worth trying actor parameter noise instead of OU action noise. 