import numpy as np
import datasets

def vec_to_int64_vec(vec):
    return np.array(vec, dtype=np.int64)

def vec_to_str(vec):
    return str(tuple(int(i) for i in vec))


def generate_example(number_of_walks = 10000, dims=10, mean=None, cov=None):
    """Problem moves: (+1, -2, +3); (); ();
    Answer: (1, 0, -1)
    """
    if mean is None:
        mean = np.zeros(dims)
    if cov is None:
        cov = np.eye(dims) * 5
    pos = np.zeros(dims)
    problem_str = ""
    for _ in range(number_of_walks):
        step_vec = vec_to_int64_vec(np.random.multivariate_normal(mean, cov))
        problem_str += vec_to_str(step_vec) + '; '
        pos += step_vec
    return problem_str, vec_to_str(pos)

tasks = []
answers = []
num_examples = 1000
number_of_walks_list = 2 + np.arange(num_examples) // 10
# dims_list = (np.arange(num_examples) % 100) + 1
# For vectorization, we need to group by dims, so let's process in batches of same dims

dims_list = (np.arange(num_examples) % 100) + 1

for dims in np.unique(dims_list):
    idxs = np.where(dims_list == dims)[0]
    n = len(idxs)
    number_of_walks = number_of_walks_list[idxs]
    # For each example in this batch, generate number_of_walks[i] steps of dims
    for i, idx in enumerate(idxs):
        n_walks = number_of_walks[i]
        # mean = np.zeros(dims)
        mean = np.random.multivariate_normal(np.zeros(dims), np.eye(dims) * 5)
        cov = np.eye(dims) * 5
        steps = np.random.multivariate_normal(mean, cov, size=n_walks)
        pos = steps.sum(axis=0)
        problem_str = '; '.join([str(tuple(int(x) for x in step.astype(np.int64))) for step in steps]) + '; '
        tasks.append(problem_str)
        answers.append(str(tuple(int(x) for x in pos.astype(np.int64))))

def save_to_huggingface(tasks, answers, dataset_name, hf_token):
    data_dict = {'task': tasks, 'answer': answers}
    ds = datasets.Dataset.from_dict(data_dict)
    ds_dict = datasets.DatasetDict({'train': ds})
    ds_dict.push_to_hub(dataset_name, token=hf_token)

if __name__ == "__main__":
    # Example usage:
    save_to_huggingface(tasks, answers, 'alexlegeartis/riw', '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')
    pass