#ifndef TSP_HPP
#define TSP_HPP

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <limits>
#include <chrono>
#include <cstdint>
#include <stdexcept>

#ifdef __linux__
#include <sys/resource.h>
#endif

struct Point {
    int id;
    double x, y;
};

struct Stats {
    std::string instance, algo, status = "OK";
    int n = 0, cost = 0, init_ub = 0;
    double time_ms = 0, time_warmup_ms = 0, time_mst_ms = 0, time_match_ms = 0, time_bnb_ms = 0;
    uint64_t expanded = 0, pruned = 0;
    long mem_kb = 0;

    void get_mem() {
#ifdef __linux__
        struct rusage u;
        if (getrusage(RUSAGE_SELF, &u) == 0) mem_kb = u.ru_maxrss;
#endif
    }
};

inline double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(high_resolution_clock::now().time_since_epoch()).count();
}

struct Graph {
    int n = 0;
    std::string name;
    std::vector<Point> pts;
    std::vector<int> dist; // Matriz plana N x N de inteiros

    void load(const std::string& path) {
        std::ifstream in(path);
        if (!in) throw std::runtime_error("Erro ao abrir " + path);
        std::string line;
        bool read_coords = false;
        while (std::getline(in, line)) {
            if (line.find("NAME") == 0) name = line.substr(line.find(':') + 1);
            else if (line.find("DIMENSION") == 0) n = std::stoi(line.substr(line.find(':') + 1));
            else if (line.find("NODE_COORD_SECTION") == 0) read_coords = true;
            else if (line.find("EOF") == 0) break;
            else if (read_coords) {
                std::istringstream iss(line);
                int id; double x, y;
                if (iss >> id >> x >> y) pts.push_back({id - 1, x, y});
            }
        }
        name.erase(0, name.find_first_not_of(" \t\r\n"));
        name.erase(name.find_last_not_of(" \t\r\n") + 1);
        n = static_cast<int>(pts.size());
        dist.assign(n * n, 0);
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                int d = static_cast<int>(std::llround(std::hypot(pts[i].x - pts[j].x, pts[i].y - pts[j].y)));
                dist[i * n + j] = dist[j * n + i] = d;
            }
        }
    }

    inline int d(int u, int v) const { return dist[u * n + v]; }

    int tour_cost(const std::vector<int>& t) const {
        int c = 0;
        for (int i = 0; i < n; ++i) c += d(t[i], t[(i + 1) % n]);
        return c;
    }
};

// --- MST (Prim O(N^2)) ---
inline std::vector<std::vector<int>> prim_mst(const Graph& g) {
    int n = g.n;
    std::vector<int> key(n, 1e9), parent(n, -1);
    std::vector<bool> in_mst(n, false);
    key[0] = 0;
    for (int i = 0; i < n; ++i) {
        int u = -1;
        for (int v = 0; v < n; ++v)
            if (!in_mst[v] && (u == -1 || key[v] < key[u])) u = v;
        in_mst[u] = true;
        for (int v = 0; v < n; ++v)
            if (!in_mst[v] && g.d(u, v) < key[v]) { key[v] = g.d(u, v); parent[v] = u; }
    }
    std::vector<std::vector<int>> adj(n);
    for (int v = 1; v < n; ++v) {
        adj[v].push_back(parent[v]);
        adj[parent[v]].push_back(v);
    }
    return adj;
}

// --- 1. Twice-Around-The-Tree ---
inline std::vector<int> twice_around(const Graph& g, Stats& st) {
    double t0 = now_ms();
    auto adj = prim_mst(g);
    st.time_mst_ms += now_ms() - t0;

    std::vector<int> tour, stack = {0};
    std::vector<bool> vis(g.n, false);
    while (!stack.empty()) {
        int u = stack.back(); stack.pop_back();
        if (!vis[u]) {
            vis[u] = true;
            tour.push_back(u);
            for (int v : adj[u]) if (!vis[v]) stack.push_back(v);
        }
    }
    return tour;
}

// --- Emparelhamento Perfeito Minimo (DP Bitmask para m<=24 + Guloso/2-opt exato rapido) ---
// Nota: Para Christofides em N<=75, o numero de vertices impares m e geralmente <= 36.
// Aqui usamos DP exata para m <= 24 e Branch-and-Bound rapido com poda para m > 24 (100% exato e < 50 linhas!)
inline void mwpm_bnb(int idx, int m, int cur_cost, int& best_cost,
                     std::vector<bool>& used, std::vector<int>& cur_match, std::vector<int>& best_match,
                     const std::vector<std::vector<int>>& w) {
    if (cur_cost >= best_cost) return;
    while (idx < m && used[idx]) idx++;
    if (idx == m) {
        best_cost = cur_cost;
        best_match = cur_match;
        return;
    }
    used[idx] = true;
    for (int j = idx + 1; j < m; ++j) {
        if (!used[j]) {
            used[j] = true;
            cur_match[idx] = j;
            cur_match[j] = idx;
            mwpm_bnb(idx + 1, m, cur_cost + w[idx][j], best_cost, used, cur_match, best_match, w);
            used[j] = false;
        }
    }
    used[idx] = false;
}

inline std::vector<std::pair<int, int>> min_weight_matching(const Graph& g, const std::vector<int>& odd) {
    int m = static_cast<int>(odd.size());
    if (m == 0) return {};
    std::vector<std::vector<int>> w(m, std::vector<int>(m, 0));
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j)
            w[i][j] = g.d(odd[i], odd[j]);

    if (m <= 22) {
        int full = (1 << m) - 1;
        std::vector<int> dp(1 << m, 1e9), pick(1 << m, 0);
        dp[0] = 0;
        for (int mask = 1; mask <= full; ++mask) {
            if (__builtin_popcount(mask) & 1) continue;
            int i = __builtin_ctz(mask);
            int rem = mask ^ (1 << i);
            for (int j = i + 1; j < m; ++j) {
                if (rem & (1 << j)) {
                    int val = dp[rem ^ (1 << j)] + w[i][j];
                    if (val < dp[mask]) { dp[mask] = val; pick[mask] = (i << 8) | j; }
                }
            }
        }
        std::vector<std::pair<int, int>> res;
        for (int cur = full; cur > 0; ) {
            int i = pick[cur] >> 8, j = pick[cur] & 0xFF;
            res.push_back({odd[i], odd[j]});
            cur ^= (1 << i) ^ (1 << j);
        }
        return res;
    }

    // Para m > 22: inicializa limite superior guloso e resolve exato via B&B
    int best_cost = 0;
    std::vector<bool> vis(m, false);
    std::vector<int> best_match(m, -1), cur_match(m, -1);
    for (int i = 0; i < m; ++i) {
        if (vis[i]) continue;
        int best_j = -1;
        for (int j = i + 1; j < m; ++j)
            if (!vis[j] && (best_j == -1 || w[i][j] < w[i][best_j])) best_j = j;
        vis[i] = vis[best_j] = true;
        best_match[i] = best_j; best_match[best_j] = i;
        best_cost += w[i][best_j];
    }
    std::fill(vis.begin(), vis.end(), false);
    mwpm_bnb(0, m, 0, best_cost, vis, cur_match, best_match, w);

    std::vector<std::pair<int, int>> res;
    for (int i = 0; i < m; ++i)
        if (best_match[i] > i) res.push_back({odd[i], odd[best_match[i]]});
    return res;
}

// --- 2. Christofides ---
inline std::vector<int> christofides(const Graph& g, Stats& st) {
    double t0 = now_ms();
    auto adj = prim_mst(g);
    st.time_mst_ms += now_ms() - t0;

    std::vector<int> odd;
    for (int i = 0; i < g.n; ++i) if (adj[i].size() % 2 == 1) odd.push_back(i);

    t0 = now_ms();
    auto pairs = min_weight_matching(g, odd);
    st.time_match_ms += now_ms() - t0;

    for (auto& p : pairs) {
        adj[p.first].push_back(p.second);
        adj[p.second].push_back(p.first);
    }

    // Circuito Euleriano (Hierholzer) + Atalho (Shortcutting)
    std::vector<int> euler, path = {0};
    while (!path.empty()) {
        int u = path.back();
        if (!adj[u].empty()) {
            int v = adj[u].back(); adj[u].pop_back();
            adj[v].erase(std::find(adj[v].begin(), adj[v].end(), u));
            path.push_back(v);
        } else {
            euler.push_back(u); path.pop_back();
        }
    }

    std::vector<int> tour;
    std::vector<bool> vis(g.n, false);
    for (int u : euler) if (!vis[u]) { vis[u] = true; tour.push_back(u); }
    return tour;
}

// --- Heuristicas de Warm-Start ---
inline std::vector<int> h_cold(const Graph& g) {
    std::vector<int> t(g.n);
    for (int i = 0; i < g.n; ++i) t[i] = i;
    return t;
}

inline std::vector<int> h_nn(const Graph& g) {
    int n = g.n, cur = 0;
    std::vector<bool> vis(n, false);
    std::vector<int> t = {0};
    vis[0] = true;
    for (int step = 1; step < n; ++step) {
        int nxt = -1;
        for (int v = 0; v < n; ++v)
            if (!vis[v] && (nxt == -1 || g.d(cur, v) < g.d(cur, nxt))) nxt = v;
        vis[nxt] = true; t.push_back(nxt); cur = nxt;
    }
    return t;
}

inline std::vector<int> h_hull(const Graph& g) {
    int n = g.n;
    if (n <= 3) return h_cold(g);
    auto pts = g.pts;
    std::sort(pts.begin(), pts.end(), [](const Point& a, const Point& b) {
        return a.x < b.x || (a.x == b.x && a.y < b.y);
    });
    auto cross = [](Point O, Point A, Point B) {
        return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);
    };
    std::vector<Point> hull;
    for (int phase = 0; phase < 2; ++phase) {
        size_t start = hull.size();
        for (int i = 0; i < n; ++i) {
            Point p = pts[phase == 0 ? i : n - 1 - i];
            while (hull.size() >= start + 2 && cross(hull[hull.size() - 2], hull.back(), p) <= 0)
                hull.pop_back();
            hull.push_back(p);
        }
        hull.pop_back();
    }

    std::vector<int> t;
    std::vector<bool> in_t(n, false);
    for (auto& p : hull) { t.push_back(p.id); in_t[p.id] = true; }

    while ((int)t.size() < n) {
        int best_v = -1, best_pos = 0, best_inc = 1e9;
        for (int v = 0; v < n; ++v) {
            if (in_t[v]) continue;
            for (int i = 0; i < (int)t.size(); ++i) {
                int u = t[i], w = t[(i + 1) % t.size()];
                int inc = g.d(u, v) + g.d(v, w) - g.d(u, w);
                if (inc < best_inc) { best_inc = inc; best_v = v; best_pos = i + 1; }
            }
        }
        t.insert(t.begin() + best_pos, best_v);
        in_t[best_v] = true;
    }
    return t;
}

// --- 3. Branch-and-Bound (Rapido e Limpo) ---
struct BnB {
    const Graph& g;
    Stats& st;
    double limit_ms, start_ms;
    int best_cost;
    std::vector<int> best_tour, cur_tour;
    std::vector<int> min1, min2;
    std::vector<std::vector<int>> adj;
    uint64_t vis_lo = 0, vis_hi = 0; // Bitmask rapida ate N=128
    bool timeout = false;

    BnB(const Graph& graph, Stats& s, double max_sec) : g(graph), st(s), limit_ms(max_sec * 1000.0) {
        int n = g.n;
        min1.assign(n, 1e9); min2.assign(n, 1e9); adj.resize(n);
        for (int i = 0; i < n; ++i) {
            std::vector<std::pair<int, int>> nb;
            for (int j = 0; j < n; ++j) {
                if (i == j) continue;
                int w = g.d(i, j);
                nb.push_back({w, j});
                if (w < min1[i]) { min2[i] = min1[i]; min1[i] = w; }
                else if (w < min2[i]) min2[i] = w;
            }
            std::sort(nb.begin(), nb.end());
            for (auto& p : nb) adj[i].push_back(p.second);
        }
    }

    inline bool is_vis(int v) const { return v < 64 ? (vis_lo >> v) & 1ULL : (vis_hi >> (v - 64)) & 1ULL; }
    inline void toggle_vis(int v) { if (v < 64) vis_lo ^= (1ULL << v); else vis_hi ^= (1ULL << (v - 64)); }

    void dfs(int lvl, int cost, int bound2) {
        if (timeout) return;
        if ((st.expanded & 16383) == 0 && (now_ms() - start_ms) >= limit_ms) {
            timeout = true; st.status = "NA"; return;
        }
        int n = g.n, u = cur_tour[lvl - 1];
        st.expanded++;

        if (lvl == n - 1) {
            for (int v : adj[u]) {
                if (!is_vis(v)) {
                    int total = cost + g.d(u, v) + g.d(v, 0);
                    if (total < best_cost) { best_cost = total; cur_tour[n - 1] = v; best_tour = cur_tour; }
                    return;
                }
            }
        }

        int m_u = (lvl == 1) ? min1[u] : min2[u];
        for (int v : adj[u]) {
            if (is_vis(v)) continue;
            int w = g.d(u, v);
            int next_b2 = bound2 + 2 * w - m_u - min1[v];
            if (next_b2 + 1 >= 2 * best_cost) { st.pruned++; continue; }

            toggle_vis(v);
            cur_tour[lvl] = v;
            dfs(lvl + 1, cost + w, next_b2);
            toggle_vis(v);
            if (timeout) return;
        }
    }

    std::vector<int> solve(const std::vector<int>& init_tour) {
        start_ms = now_ms();
        best_tour = init_tour;
        best_cost = st.init_ub = g.tour_cost(init_tour);
        int b2 = 0;
        for (int i = 0; i < g.n; ++i) b2 += min1[i] + min2[i];

        cur_tour.assign(g.n, 0);
        toggle_vis(0);
        dfs(1, 0, b2);
        st.time_bnb_ms = now_ms() - start_ms;
        return best_tour;
    }
};

#endif
