import tensorflow as tf
from neuralnetwork import NeuralNetwork
from layers import ScalarMultiplyLayer, AdditionLayer


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
        actor.set_input_layer(0, state_input)
        actors.append(actor)
        action_pred = actor.get_output_layer()

        if create_intermediate_value_networks:
            value = value_network.copy(name + "_value_" + str(step), reuse_parameters=True)
            value.set_input_layer(0, current_state)
            values.append(value)

        model = model_network.copy(name + "_model_" + str(step), reuse_parameters=True)
        model.set_input_layer(0, state_input)
        model.set_input_layer(1, action_pred)
        models.append(model)
        next_state_pred = model.get_output_layer()
        current_state = next_state_pred

        reward = reward_network.copy(name + "_reward_" + str(step), reuse_parameters=True)
        reward.set_input_layer(0, state_input)
        reward.set_input_layer(1, action_pred)
        rewards.append(reward)
        reward_pred = reward.get_output_layer()

        if step == 0:
            discounted_rewards_sum_pred = reward_pred
        else:
            discounted_reward_pred = ScalarMultiplyLayer("discounted_reward_" + str(step), reward_pred,
                                                         pow(discount_factor, step + 1))
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

