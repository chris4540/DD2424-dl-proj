import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("progressive_block_exp.csv")
    df = df[df['epoch'] == 10]
    df = df[df['alpha'] < 2]
    cols = ['alpha', 'test', 'validation']
    df2 = df[cols]
    df2 = df2.set_index('alpha')
    df2 = df2.sort_index()

    ax = df2.plot(kind='line', style='x-')
    ax.set_ylabel("Accuracy")
    ax.set_ylim([0.75, 0.95])
    fig = ax.get_figure()
    fig.savefig("pbt_alpha_acc.png", bbox_inches='tight')

