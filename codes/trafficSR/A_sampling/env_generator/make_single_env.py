import codes.trafficSR.A_sampling.env_generator.SRenv as SRenv


def divide_number_into_parts(total, n):
    # 计算每份的基本大小和余数
    part_size = total // n
    remainder = total % n

    # 创建一个列表，每个元素是每份的大小
    parts = [part_size] * n

    # 将余数平均分配给列表的前几个元素
    for i in range(remainder):
        parts[i] += 1

    return parts


def make_env_fn(X_numpy,
                y_numpy,
                train_args=None,
                token_args=None,
                ngsim_args=None,
                prefix_tree=None,
                seed=100,
                batch_size=None):
    env = SRenv.SRenv(
        X_numpy=X_numpy,
        y_numpy=y_numpy,
        library_args=train_args['library_args'],
        env_args=train_args['env_args'],
        token_args=token_args,
        ngsim_args=ngsim_args,
        bool_args=train_args['bool_args'],
        prefix_tree=prefix_tree,
        seed=seed,
        batch_size=batch_size,
    )
    return env


def get_make_env_kargs(**kargs):
    return kargs
