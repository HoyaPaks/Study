import numpy as np
from matplotlib import pyplot as plt

# from JSAnimation.IPython_display import display_animation
from matplotlib import animation
# from IPython.display import HTML
from IPython import display
import torch


def running_mean(x, N=50):
    kernel = np.ones(N)
    conv_len = x.shape[0]-N
    y = np.zeros(conv_len)
    for i in range(conv_len):
        y[i] = kernel @ x[i:i+N]
        y[i] /= N
    return y


def display_frames_as_gif(frames):
    """
    Displays a list of frames as a gif, with controls
    """
    patch = plt.imshow(frames[0][0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i][0])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=10)
    
    video = anim.to_html5_video()
    html = display.HTML(video)
    display.display(html)
    plt.close()


def minigrid_test_model(env, model, mode='static', loss_lim=15, display=True):
    action_set = {
        0: 'u',
        1: 'd',
        2: 'l',
        3: 'r',
    }

    i = 0
    test_game = env(mode=mode)
    state = test_game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
    state = torch.from_numpy(state).float()

    if display:
        print("Initial State:")
        print(test_game.display())

    status = 1
    while status == 1:
        qval = model(state)
        qval_ = qval.data.numpy()
        action_ = np.argmax(qval_)
        action = action_set[action_]

        if display:
            print('Move #: %s; Taking action: %s' % (i, action))
        test_game.makeMove(action)

        state_ = test_game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
        state = torch.from_numpy(state_).float()

        if display:
            print(test_game.display())
        reward = test_game.reward()
        if reward != -1:
            if reward > 0:
                status = 2
                if display:
                    print("Game won! Reward: %s" % (reward,))
            else:
                status = 0
                if display:
                    print("Game LOST. Reward: %s" % (reward,))
        i += 1

        if i > loss_lim:
            if display:
                print("Game lost; too many moves.")
            break

    win = True if status == 2 else False
    return win


def cartpole_train_graph(score):
    score = np.array(score)
    avg_score = running_mean(score, 50)
    plt.figure(figsize=(10, 7))
    plt.ylabel("Episode Duration", fontsize=22)
    plt.xlabel("Training Epochs", fontsize=22)
    plt.plot(avg_score, color='green')


def cartpole_test_scatter(env, model, MAX_DUR=500):
    score = []
    games = 100
    done = False
    state1 = env.reset()
    for i in range(games):
        t = 0
        while not done:
            pred = model(torch.from_numpy(state1).float())
            action = np.random.choice(np.array([0, 1]), p=pred.data.numpy())
            state2, reward, done, truncated = env.step(action)
            state1 = state2
            t += 1
            if t > MAX_DUR:
                break

        state1 = env.reset()
        done = False
        score.append(t)
    score = np.array(score)
    plt.scatter(np.arange(score.shape[0]), score)


def cartpole_test_scatter_a2c(env, model, MAX_DUR=500):
    score = []
    games = 100
    done = False
    state1 = env.reset()

    for i in range(games):
        t = 0
        while not done:
            logits, value = model(torch.from_numpy(state1).float())
            action_dist = torch.distributions.Categorical(logits=logits)
            action = action_dist.sample()
            state2, reward, done, truncated = env.step(action.detach().numpy())
            state1 = state2

            t += 1
            if t > MAX_DUR:
                break

        state1 = env.reset()
        done = False
        score.append(t)

    score = np.array(score)
    plt.scatter(np.arange(score.shape[0]), score)


def cartpole_test_video(env, model, MAX_DUR=500):
    done = False
    frames = []
    t = 0
    state1 = env.reset()
    while not done:
        frames.append(env.render())
        pred = model(torch.from_numpy(state1).float())
        action = np.random.choice(np.array([0, 1]), p=pred.data.numpy())
        state2, reward, done, truncated = env.step(action)
        state1 = state2
        t += 1
        if t > MAX_DUR:  # L
            break

    env.close()
    print(f"Episode length : {t}")

    display_frames_as_gif(frames)


def cartpole_test_video_a2c(env, model, MAX_DUR=500):
    done = False
    frames = []
    t = 0
    state1 = env.reset()

    while not done:
        frames.append(env.render())
        logits, value = model(torch.from_numpy(state1).float())
        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample()
        state2, reward, done, truncated = env.step(action.detach().numpy())
        state1 = state2
        t += 1
        if t > MAX_DUR:  # L
            break

    env.close()
    print(f"Episode length : {t}")

    display_frames_as_gif(frames)