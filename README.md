# RumorSphere: A Framework for Million-scale Agent-based Dynamic Simulation of Rumor Propagation

# 🤖 Introduction
RumorAgent is a novel dynamic and hierarchical social network simulation framework designed to model large-scale rumor propagation in social networks. RumorAgent constructs a simulated Twitter environment, where agents can perform actions such as posting, commenting, and retweeting. Through a multi-agent dynamic interaction strategy, we effectively combine agent-based models and LLMs, dynamically determining agent types in each round to improve simulation efficiency. The framework uses a hierarchical collaborative network to build more realistic relationship topologies. Additionally, RumorAgent offers counterfactual experimentation capabilities to explore various hypothetical scenarios, providing valuable insights for policymaking and public opinion management.

# 💡 Features
* **Scalability:** Supports simulations with millions of agents, dynamically adjusting to large-scale networks.

* **Counterfactual Experiments:** Simulate hypothetical scenarios and causal relationships for intervention strategies.

* **Multi-agent Dynamic Interaction Strategy:** : Differentiates between core and ordinary agents dynamically for efficient simulation.

* **Hierarchical Collaborative Network:** Enhances local community structures, fostering faster rumor propagation.

# 🚀 Getting Started

## Requirements and Installation
**Make sure you have Python >= 3.9**

1. Clone the repository:
    ```shell
    git clone <repository-url>
    ```

2. Install Conda environment：
    ```shell
    conda create -n RumorSphere python=3.9
    conda activate RumorSphere
    pip install -e .
    ```

3. Set your OpenAI API key:
    ```shell
    # Export your OpenAI API key
    export OPENAI_API_BASE="your_api_base_here"
    export OPENAI_API_KEY="your_api_key_here"
    ```

4. You can adjust other parameters in  `agentverse/tasks/simulation/config.yaml`.

## Running the Simulation

### CLI Demo
To start a simulation using the Command Line Interface (CLI), execute the following command:
   ```shell
   python -u simulator.py --config_file agentverse/tasks/simulation/config.yaml --output_file messages.json --log_file simulation.log
   ```
This command will initialize the simulation with the configuration found in agentverse/tasks/simulation/config.yaml, output the messages to messages.json, and log the details of the simulation in simulation.log.


# 🤝 Contributing
We welcome contributions from the community. If you'd like to contribute, please fork the repository and use a feature branch. Pull requests are warmly welcome.

# 📝 License
RumorAgent is distributed under the [MIT License](./LICENSE). You are free to use and modify it for your own projects.

# 📄 Cite
More details will be released soon.
<!-- If you use RumorAgent in your research, please cite our work as follows:
   ```shell
   More details will be released soon.
   ``` -->