from time import sleep

from bit_inversion_envs import VanillaBitInversionEnv
from LLMAgents import ChatGPT35Agent, Davinci3Agent


env = VanillaBitInversionEnv(n_bits=4)
agent = ChatGPT35Agent(num_actions=env.action_space.n, state_size=env.observation_space.n)

state = env.reset(start_state=[0, 0, 0, 0])
action = agent.reset_agent(state)
for i in range(20):
    state, reward, done, truncated, info = env.step(action)
    action = agent.step(state, reward)
    # wait 20 seconds to avoid rate limit
    sleep(20)

print(f"used {agent.tokens_used} tokens")
