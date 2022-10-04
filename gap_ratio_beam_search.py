def gap_ratio_beam_search(env, initial_state, num_steps, beam_size, branching, subsampling):

    spec = env.action_spec()
    states = [initial_state[np.newaxis,:]]
    actions = [None]

    for t in range(num_steps):
        print(f"step {t} of {num_steps}...")

        P = len(states[t])
        child_states = np.empty((P, branching, 2), dtype=np.float32)
        if t == 0 or not do_walk:
            child_actions = rng.uniform(-1, 1, (P, branching, 1)).astype(np.float32)
        else:
            child_actions = np.random.randn(P, branching, 1).astype(np.float32) * walk_stdev
            child_actions += actions[t][:, np.newaxis, :]

        for p in range(P):
            for b in range(branching):
                env.state = states[t][p].copy()
                child_states[p,b], _, _, _ = env.step(np.clip(child_actions[p,b], -1, 1))
        child_states = child_states.reshape(-1, 2)
        child_actions = child_actions.reshape(-1, 1)

        if len(child_states) < beam:
            states.append(child_states)
            actions.append(child_actions)
            continue

        # set-difference farthest point algorithm
        # modified from
        #    https://doi.org/10.1137/15M1051300 | https://arxiv.org/pdf/1411.7819.pdf
        #    https://doi.org/10.1016/0304-3975(85)90224-5

        previous_states = np.concatenate(states, axis=0)
        if len(previous_states) > sampling:
            subsample = np.random.choice(len(previous_states), size=sampling, replace=False)
            previous_states = previous_states[subsample]
        bkgd_states = np.concatenate((child_states, previous_states), axis=0)

        dists = np.linalg.norm(child_states[:,np.newaxis,:] - bkgd_states[np.newaxis,:,:], axis=2)
        included = np.ones(len(bkgd_states), dtype=bool)
        included[:len(child_states)] = False
        excluded = list(range(len(child_states)))

        a = dists[:,included].min(axis=1).argmax()
        included[excluded.pop(a)] = True
        
        for p in range(1, beam):
            a = dists[excluded][:,included].min(axis=1).argmax()
            included[excluded.pop(a)] = True

        new_actions = child_actions[included[:len(child_states)]]
        new_states = child_states[included[:len(child_states)]]
        states.append(new_states)
        actions.append(new_actions)

        # pt.cla()
        # draw(states)
        # # input('.')

    env.close()


