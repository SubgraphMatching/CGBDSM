#include <tuple>
#include "utils/nucleus/nd.h"

// from boost (functional/hash):
// see http://www.boost.org/doc/libs/1_35_0/doc/html/hash/combine.html template
/*template <typename T>
inline void hash_combine(std::size_t &seed, const T &val) {
    seed ^= std::hash<T>()(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}*/

// auxiliary generic functions to create a hash value using a seed
template <typename T>
inline void hash_val(std::size_t &seed, const T &val) {
    hash_combine(seed, val);
}

template <typename T, typename... Types>
inline void hash_val(std::size_t &seed, const T &val, const Types &... args) {
    hash_combine(seed, val);
    hash_val(seed, args...);
}

template <typename... Types>
inline std::size_t hash_val(const Types &... args) {
    std::size_t seed = 0;
    hash_val(seed, args...);
    return seed;
}

struct pair_hash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2> &p) const {
        return hash_val(p.first, p.second);
    }
};

struct tuple_hash {
    // Recursive template code derived from Matthieu M.
    template <class Tuple, size_t Index = std::tuple_size<Tuple>::value - 1>
    struct Impl{
        static void apply(size_t& seed, Tuple const& tuple)
        {
            Impl<Tuple, Index-1>::apply(seed, tuple);
            hash_combine(seed, std::get<Index>(tuple));
        }
    };

    template <class Tuple>
    struct Impl<Tuple, 0>{
        static void apply(size_t& seed, Tuple const& tuple)
        {
            hash_combine(seed, std::get<0>(tuple));
        }
    };

    template <class Tuple>
    inline std::size_t hash_val(Tuple const&p) const {
        std::size_t seed = 0;
        Impl<Tuple>::apply(seed, p);
        return seed;
    }

    template <typename... TT>
    std::size_t operator()(const std::tuple<TT...> &p) const {
        return hash_val(p);
    }
};
