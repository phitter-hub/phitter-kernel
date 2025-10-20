async function getPhitterContinuousCode() {
    const response = await fetch("../../../../phitter_web/discrete/phitter_web_discrete.py");
    const discreteCode = await response.text();
    return discreteCode;
}

async function main() {
    console.time("Total Time");
    
    const startTime = performance.now();

    const pyodide = await loadPyodide({
        indexURL: "https://cdn.jsdelivr.net/pyodide/v0.25.1/full",
    });
    await pyodide.loadPackage(["scipy"]);

    const endTime = performance.now();
    const loadTimeInSeconds = (endTime - startTime) / 1000;
    console.log(`Pyodide and SciPy Load Time: ${loadTimeInSeconds.toFixed(2)} seconds`);

    const discreteCode = await getPhitterContinuousCode();

    console.time("Python Code Execution Time");

    await pyodide.runPython(discreteCode);

    pyodide.runPython(`
        import time
        data = BINOMIAL(init_parameters_examples=True).sample(100000)

        ti = time.time()
        print("Init")
        phitter_discrete = PHITTER_DISCRETE(data=data)
        phitter_discrete.fit()
        tf = time.time()
        print(f"Execution time: {tf - ti}")

        sorted_distributions = phitter_discrete.sorted_distributions
        not_rejected_distributions = phitter_discrete.not_rejected_distributions

        for distribution, results in not_rejected_distributions.items():
            print(f"Distribution: {distribution}, SSE: {results['sse']}, Aprobados: {results['n_test_passed']}")
    `);

    console.timeEnd("Python Code Execution Time");
    console.timeEnd("Total Time");
}

main();
