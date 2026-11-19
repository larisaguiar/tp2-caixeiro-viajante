#include <iostream>
#include <iomanip>
#include "tsp.hpp"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Uso: ./tsp <instancia.tsp> <algo> [timeout_seg]\n"
                  << "Algos: twice_around, christofides, bnb_cold, bnb_nn, bnb_hull, bnb_twice, bnb_christofides\n";
        return 1;
    }

    Graph g;
    g.load(argv[1]);
    std::string algo = argv[2];
    double timeout = (argc >= 4) ? std::stod(argv[3]) : 1800.0;

    Stats st;
    st.instance = g.name;
    st.n = g.n;
    st.algo = algo;

    double t0 = now_ms();
    std::vector<int> tour;

    if (algo == "twice_around") tour = twice_around(g, st);
    else if (algo == "christofides") tour = christofides(g, st);
    else if (algo.find("bnb_") == 0) {
        double tw = now_ms();
        if (algo == "bnb_cold") tour = h_cold(g);
        else if (algo == "bnb_nn") tour = h_nn(g);
        else if (algo == "bnb_hull") tour = h_hull(g);
        else if (algo == "bnb_twice") tour = twice_around(g, st);
        else if (algo == "bnb_christofides") tour = christofides(g, st);
        st.time_warmup_ms = now_ms() - tw;

        BnB solver(g, st, timeout);
        tour = solver.solve(tour);
    }

    st.cost = g.tour_cost(tour);
    st.time_ms = now_ms() - t0;
    st.get_mem();

    std::cout << st.instance << "," << st.n << "," << st.algo << "," << st.status << ","
              << st.cost << "," << st.init_ub << "," << std::fixed << std::setprecision(4)
              << st.time_ms << "," << st.time_warmup_ms << "," << st.time_mst_ms << ","
              << st.time_match_ms << "," << st.time_bnb_ms << "," << st.expanded << ","
              << st.pruned << "," << st.mem_kb << "\n";
    return 0;
}
