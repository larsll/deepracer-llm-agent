from setuptools import setup, find_packages

setup(
    name="deepracer_llm_agent",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "boto3>=1.28.0",
        "requests",
        "numpy>=1.24.0",
        "pillow>=10.0.0",
        "python-dotenv>=1.0.0",
    ],
    description="DeepRacer LLM Agent for autonomous driving using large language models",
    author="AWS DeepRacer",
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "deepracer-llm-agent=deepracer_llm_agent.__main__:main",
        ],
    },
)