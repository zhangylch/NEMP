import cuequivariance as cue

from low_level import cueq_tp


def _old_even_rule_counts(rmaxl, prmaxl):
    count_l = [0] * prmaxl
    for lf in range(prmaxl):
        for li1 in range(rmaxl):
            low = abs(li1 - lf)
            up = min(prmaxl, li1 + lf + 1)
            for li2 in range(low, up):
                if (li1 + li2 + lf) % 2 == 0:
                    count_l[lf] += 1
    return count_l


def _so3_path_counts(rmaxl, prmaxl):
    def irreps(max_l):
        return cue.Irreps("SO3", " + ".join(f"1x{l}" for l in range(max_l)))

    poly = cue.descriptors.fully_connected_tensor_product(
        irreps(rmaxl),
        irreps(prmaxl),
        irreps(prmaxl),
    )
    stp = poly.polynomial.operations[0][1]
    count_l = [0] * prmaxl
    for path in stp.paths:
        count_l[path.indices[3]] += 1
    return count_l


def test_o3_natural_parity_matches_previous_even_path_rule():
    rmaxl, prmaxl = 4, 4

    _, o3_count_l = cueq_tp.tensor_product_path_metadata(rmaxl, prmaxl, nwave=1)
    old_count_l = _old_even_rule_counts(rmaxl, prmaxl)
    so3_count_l = _so3_path_counts(rmaxl, prmaxl)

    assert list(map(int, o3_count_l)) == old_count_l
    assert so3_count_l != old_count_l
