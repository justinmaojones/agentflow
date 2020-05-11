from agentflow.examples import pong_ddpg_stable
import click

if __name__=='__main__':
    click.command()(pong_ddpg_stable.run)()
