import sys

sys.path.append("../../../phitter_web/continuous/")
from phitter_web_continuous import PhitterContinuous


def get_data(path: str) -> list[float | int]:
    sample_distribution_file = open(path, "r")
    data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]
    sample_distribution_file.close()
    return data


if __name__ == "__main__":
    path = "../../../datasets_test/continuous/data_1000/data_alpha.txt"
    # path = "../datasets_test/discrete/book2.txt"
    # path = "../datasets_test/continuous/data_1000/data_beta.txt"

    data = get_data(path)

    phi = PhitterContinuous(data)
    phi.fit(n_workers=2)

    for distribution, results in phi.sorted_distributions.items():
        print(f"Distribution: {distribution}, SSE: {results['sse']}, Aprobados: {results['n_test_passed']}")
