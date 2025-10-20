async function getPhitterContinuousCode() {
    const resonse = await fetch("../../../../phitter_web/continuous/phitter_web_continuous.py");
    const continuousCode = await resonse.text();
    return continuousCode;
}

async function main() {
    console.time("Total Time");
    console.time("Pyodide Load Time");

    let pyodide = await loadPyodide({
        indexURL: "https://cdn.jsdelivr.net/pyodide/v0.26.1/full",
    });


    console.timeEnd("Pyodide Load Time");
    console.time("SciPy Load Time");

    await pyodide.loadPackage(["scipy"]);

    console.timeEnd("SciPy Load Time");
    const continuousCode = await getPhitterContinuousCode();

    console.time("Python Code Execution Time");

    pyodide.runPython(continuousCode);

    pyodide.runPython(`
        import sys
        print(sys.version)

        import scipy
        print(scipy.__version__)

        import time
        data = BETA(init_parameters_examples=True).sample(10000)

        ti = time.time()
        print("Init")
        phitter_continuous = PHITTER_CONTINUOUS(data=data, subsample_estimation_size=10000)
        phitter_continuous.fit()
        tf = time.time()
        print(f"Execution time: {tf - ti}")

        sorted_distributions = phitter_continuous.sorted_distributions
        not_rejected_distributions = phitter_continuous.not_rejected_distributions

        for distribution, results in not_rejected_distributions.items():
            print(f"Distribution: {distribution}, SSE: {results['sse']}, Aprobados: {results['n_test_passed']}")
    `);

    console.timeEnd("Python Code Execution Time");
    console.timeEnd("Total Time");
}

main();
