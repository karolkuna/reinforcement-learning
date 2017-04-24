import tensorflow as tf
from neuralnetwork import NeuralNetwork
from layers import ScalarMultiplyLayer, AdditionLayer, SubtractionLayer, SquaredDifferenceLayer, SumLayer


def create_actor_critic_network(name, actor_network, critic_network):
    actor_critic = NeuralNetwork(name, actor_network.session, actor_network.input_dims)

    actor = actor_network.copy(name + "_actor", reuse_parameters=True)
    actor.set_input_layer(0, actor_critic.get_input_layer(0))

    critic = critic_network.copy(name + "_critic", reuse_parameters=True)
    critic.set_input_layer(0, actor_critic.get_input_layer(0))
    critic.set_input_layer(1, actor.get_output_layer())

    actor_critic.compile(critic.get_output_layer())
    return actor_critic, actor, critic


def create_actor_model_critic_network(name, actor_network, model_network, reward_network, value_network, discount_factor, forward_steps, create_intermediate_value_networks=False):
    actor_model_critic = NeuralNetwork(name, actor_network.session, actor_network.input_dims)
    state_input = actor_model_critic.get_input_layer(0)
    discounted_rewards_sum_pred = None
    current_state = state_input

    actors = []
    models = []
    rewards = []
    values = []

    for step in range(forward_steps):
        actor = actor_network.copy(name + "_actor_" + str(step), reuse_parameters=True)
        actor.set_input_layer(0, current_state)
        actors.append(actor)
        action_pred = actor.get_output_layer()

        if create_intermediate_value_networks:
            value = value_network.copy(name + "_value_" + str(step), reuse_parameters=True)
            value.set_input_layer(0, current_state)
            values.append(value)

        reward = reward_network.copy(name + "_reward_" + str(step), reuse_parameters=True)
        reward.set_input_layer(0, current_state)
        reward.set_input_layer(1, action_pred)
        rewards.append(reward)
        reward_pred = reward.get_output_layer()

        model = model_network.copy(name + "_model_" + str(step), reuse_parameters=True)
        model.set_input_layer(0, current_state)
        model.set_input_layer(1, action_pred)
        models.append(model)
        next_state_pred = model.get_output_layer()
        current_state = next_state_pred

        if step == 0:
            discounted_rewards_sum_pred = reward_pred
        else:
            discounted_reward_pred = ScalarMultiplyLayer("discounted_reward_" + str(step), reward_pred,
                                                         pow(discount_factor, step))
            discounted_rewards_sum_pred = AdditionLayer("discounted_rewards_sum_" + str(step),
                                                        [discounted_rewards_sum_pred, discounted_reward_pred])

    value = value_network.copy(name + "_value_" + str(forward_steps), reuse_parameters=True)
    value.set_input_layer(0, current_state)
    values.append(value)
    value_pred = value.get_output_layer()

    discounted_value_pred = ScalarMultiplyLayer("discounted_value", value_pred,
                                                pow(discount_factor, forward_steps))
    expected_return = AdditionLayer("expected_return", [discounted_value_pred, discounted_rewards_sum_pred])

    actor_model_critic.compile(expected_return, [net.get_output_layer() for net in values[:-1]])
    return actor_model_critic, actors, models, rewards, values


def create_model_based_td_error_network(name, actor_network, model_network, reward_network, value_network, discount_factor):
    td_error_network = NeuralNetwork(name, actor_network.session, value_network.input_dims)
    state_input = td_error_network.get_input_layer(0)

    value = value_network.copy(name + "_value", True)
    value.set_input_layer(0, state_input)
    value_pred = value.get_output_layer()

    actor = actor_network.copy(name + "_actor", True)
    actor.set_input_layer(0, state_input)
    action_pred = actor.get_output_layer()

    reward = reward_network.copy(name + "_reward", True)
    reward.set_input_layer(0, state_input)
    reward.set_input_layer(1, action_pred)
    reward_pred = reward.get_output_layer()

    model = model_network.copy(name + "_model", True)
    model.set_input_layer(0, state_input)
    model.set_input_layer(1, action_pred)
    next_state_pred = model.get_output_layer()

    next_value = value_network.copy(name + "_next_value", True)
    next_value.set_input_layer(0, next_state_pred)
    next_value_pred = next_value.get_output_layer()
    discounted_value_pred = ScalarMultiplyLayer("discounted_value", next_value_pred, discount_factor)
    expected_return = AdditionLayer("expected_return", [discounted_value_pred, reward_pred])
    td_error = SubtractionLayer("td_error", [expected_return, value_pred])

    td_error_network.compile(td_error)
    return td_error_network


def create_squared_error_network(name, network):
    input_dims = list(network.input_dims)
    output_dim = network.get_output_layer().get_size()
    input_dims.append(output_dim)  # target input
    error_network = NeuralNetwork(name, network.session, input_dims)
    target = error_network.get_input_layer(len(input_dims) - 1)

    network_copy = network.copy(name + "_" + network.name, True)
    for i in range(len(network.input_dims)):
        network_copy.set_input_layer(i, error_network.get_input_layer(i))
    output_pred = network_copy.get_output_layer()

    squared_diff = SquaredDifferenceLayer("squared_diff", [output_pred, target])

    if output_dim == 1:
        squared_error_sum = squared_diff
    else:
        squared_error_sum = SumLayer("squared_error_sum", squared_diff)

    error_network.compile(squared_error_sum)
    return error_network
