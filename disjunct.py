import time
import math
import numpy as np
import unireedsolomon as rs
import matplotlib.pyplot as plt


def decode(m, x):
    # Description: decodes x out of M and r when r=M*x.
    # Input:
        # t by n binary matrix M of type np.matrix,
        # vector r with length t of type np.array
    # Output:
        # binary vector x of type np.array
    # running time: O(nt)

    result = np.prod(x | ~m.T, axis=1)
    return np.array(result, dtype=bool)


def generate_vector(n, d):
    # Description: generates a vector of size n with d ones (Trues)
    # Input: n,d of type int
    # Output: vector x of size n of type np.array
    # running time: O(n)
    x = np.zeros(n, dtype=bool)
    x[np.random.choice(n, d, False)] = np.ones(d, dtype=bool)  # replaces d indices from False to True (0 to 1)
    return x


#Utils
def validate_matrix_shape(m, m_expected_shape, name):
    # Description: validate matrix shape - used for debugging
    # Input:
        # m - Matrix
        # m_expected_shape - Matrix expected shape
        # name - Matrix name
    # Output: None
    # running time: O(1)
    x = np.asarray(list(m))
    if isinstance(m_expected_shape, int):
        m_expected_shape = tuple((m_expected_shape,))
    if x.shape != m_expected_shape:
        raise ValueError(name + "'s shape is: ", x.shape, "expected: ", m_expected_shape)


def bin_list_to_costume_list(x, base):
    # Description: Convert binary list to 'base' list(vector with elements in GF(2**base))
    # Input:
        # x - Binary vector
    # Output:
        # Base vector
    # running time: O(n)
    return [int(''.join(map(lambda a: str(int(a)), x[n:n+base])), 2) for n in range(0, len(x), base)]


def compare_matricies(a, b, names):
    a = np.asarray(a)
    b = np.asarray(b)
    same = np.sum(a == b)
    different = a.size - same
    ratio = same / (different + same)
    s = '''
    ******************\n
    Compares {name_1} and {name_2} \n
    Same: {s} \n
    Different: {d}\n
    Ratio: {r} \n
    Are equal: {equal}\n
    ******************\n
    '''.format(name_1=names[0], name_2=names[1], s=same, d=different, r=ratio, equal=np.array_equal(a, b))
    return s


#Concatenated code
def create_cout_encoder(n1, k1, k2):
    # Description: Create [n1 X k1] GF(2**k2) matrix
    # Input:
        # n1 - Rows number
        # k1 - Columns number
        # k2 - log(q), q - alphabet size
    # Output:
        # A random [n1 X k1] binary matrix with prob of entry being 1 is 1 / (10 * d)
    generator = 3
    poly = rs.find_prime_polynomials(generator, k2, single=True)
    coder = rs.RSCoder(n1, k1, generator, poly, 1, k2)
    return coder


def generate_inner_code(n2, k2, d):
    # Description: Generate random [n2 X k2] binary matrix
    # Input:
        # n2 - Rows number
        # k2 - Columns number
        # d - disjunctness
    # Output:
        # A random [n2 X k2] binary matrix with prob of entry being 1 is 1 / (10 * d)
    # Running Time: O(n2*k2)
    prob = 1 / (10 * d)
    return np.array(np.random.choice((0, 1), (n2, k2), p=[1 - prob, prob]), dtype=bool)


def concatinate_codes(cout, cin_list, x, n2, n1, k2, k1, d):
    # Description: Apply Cout o Cin(x)
    # Input:
        # cout - [n1 X k1] matrix over GF(2**k2) cout of type np.array
        # cin_list - n1 [n2 X k2] binary matrices Cin's of type np.array
        # x - [k1 X 1] GF(2**k2) vector of type np.array
        # Rest of the params are defined constants in the paper
    # Output: n2*n1 X 1 binary vector of type np.array
    # Running Time: O(n2*n1*k2)     ~O(log**3(n))
    # Memory:       O(n2*n1)        ~O(log**2(n))

    #Apply Cout(x)
    #Shape: [n1 X 1] each entry in 2 ** k2
    #Running Time: O(enc)
    #Memory: O(n1)              ~O(log**2(n))
    cout_x = cout.encode_fast(x, return_string=False)

    #Validates C(x) size
    validate_matrix_shape(cout_x, n1, "Cout(x)")

    # Apply Cin's on Cout(x)
    # [n2n1 X 1] each entry in 2
    # Running Time: O(n1*n2*k2)     ~O(log**3(n))
    # Memory: O(n1*n2)              ~O(log**2(n))
    l = []
    for i, value in enumerate(cout_x):
        # 2 ** k2 entry to [k2 X 1] binary vector transformation
        binary_value = np.array(list(bin(value)[2:].zfill(k2))).astype(bool)
        z = list(cin_list[i].dot(binary_value))
        l.append(z)

    r = sum(l, [])

    # Validates C(x) size
    validate_matrix_shape(r, n1*n2, "Cout o Cin_list(x)")

    return r


def apply_concatinated_codes(k1, k2, n1, n2, d, x):
    # Description: Build Cout and Cin_list and apply Cout o List_Cin's(x)
    # Input:
        # x - Vector to encode
        # Rest of the params are defined constants in the paper
    # Output:
        # [n2*n1 X 1] binary vector of type np.array
    # Running Time: O(n2*n1*k2)     ~O(log**3(n))
    # Memory:       O(n2*n1)        ~O(log**2(n))

    if n1 > 2 ** k2:
        raise ValueError('n1 has to be smaller than 2**k2, n1: ', n1, " 2**k2: ", 2 ** k2)

    # Create Cout encoder
    cout = create_cout_encoder(n1, k1, k2)

    #Generate n1 Cins of shape: [n2 X k2]
    #Running Time: O(n1*(n2*k2))   ~O(log**3(n)
    # Memory:      O(n1*(n2*k2))   ~O(log**3(n)
    inner_codes = [generate_inner_code(n2, k2, d) for i in range(n1)]
    #Validates Cin's shape
    cin = inner_codes[0]
    validate_matrix_shape(cin, (n2, k2), "Cin")
    if len(inner_codes) != n1:
        raise ValueError("Cin list's size is: ", len(inner_codes), 'expected: ', n1)

    #                                n1 * n2, 2 ** (k1 * k2)
    # Apply concatinated codes on message t X n
    # Running Time: O(n2*n1*k2)     ~O(log**3(n))
    # Memory:       O(n2*n1)        ~O(log**2(n))
    r = concatinate_codes(cout, inner_codes, x, n2, n1, k2, k1, d)

    return r, cout, inner_codes


def apply_decoding(k1, k2, n1, n2, d, r, cout, cins_list):
    # Description: Apply DeCout o List_DeCin's(x)
    # Input:
        # x - [n2*n1 X 1]Vector to decode
        # Cout - cout encoder
        # Cins_list - List of cin encoders
        # Rest of the params are defined constants in the paper
    # Output:
        # [k1 X 1] GF(2**k2) vector of type np.array
    # Running Time: O(n1*(n2*k2))   ~O(log**3(n))
    # Memory:       O(n1)           ~O(log(n))

    # Split [n2*n1 X 1] binary list to [n1 X n2] binary list
    r_decodeable = [r[i:i + n2] for i in range(0, len(r), n2)]

    # Decode cin's
    # Running Time: O(n1*(n2*k2))     ~O(log**3(n))
    # Memory:       O(n1)             ~O(log(n))
    cins_decoded = []
    for r_i, cin_i in zip(r_decodeable, cins_list):
        cin_decoded = list(decode(cin_i, r_i))
        cins_decoded.append(list(map(lambda a: int(str(int(a)), 2), cin_decoded)))
    # Validates cins_decoded's shape
    validate_matrix_shape(cins_decoded, (n1, k2), "Decoded Cins")

    # Convert nested np.array to nested list
    cins_decoded_list = list(map(lambda a: list(a), np.asarray(cins_decoded)))

    # Reshapes cins_decoded_list [n1 X k2] to [n1, 1] GF(2**k2)
    # running time: O(n1)   ~O(log(n))
    computed_cout_encoding = bin_list_to_costume_list(sum(cins_decoded_list, []), k2)
    # Validates computed_cout_encoding's shape
    validate_matrix_shape(computed_cout_encoding, n1, "computed_cout_encoding")

    # Decode Cout
    cout_decoding = cout.decode_fast(computed_cout_encoding, return_string=False)
    decoding = cout_decoding[0]

    return decoding


def check_code_correctness(k1, d, k2, is_ui):
    # Description: Measures the time it takes to encode/decode random message
    # Input:
        # Paper params
    # Output:
        # Success status
        # Encoding measured time
        # Decoding measured time
    # Running Time: ~O(log**3(n))
    # Memory:       ~O(log**2(n))

    # Proportions are taken from the paper
    n1 = 10 * d * k1
    n2 = 480*d * k2
    t = n1 * n2
    n = 2 ** (k1 * k2)
    q = 2 ** k2
    if is_ui:
        print('k1: ', k1)
        print('k2: ', k2)
        print('d: ', d)
        print('n1: ', n1)
        print('n2: ', n2)
        print('t: ', t)
        print('n: ', n)
        print('q: ', q)

    # Generate binary message [k2*k1 X 1] to encode
    binary_message = generate_vector(k2*k1, d)
    # Split message to [k2 X k1] nested list
    message = bin_list_to_costume_list(binary_message, k2)

    # Time encoding
    encoding_t0 = time.time()
    # Apply Cout o Cin's(message)
    # Running Time: O(n2*n1*k2)     ~O(log**3(n))
    # Memory:       O(n2*n1)        ~O(log**2(n))
    r, cout, cins_list = apply_concatinated_codes(k1, k2, n1, n2, d, message)
    encoding_t = time.time() - encoding_t0

    # Validates r's shape
    validate_matrix_shape(r, t, "r")

    # Time decoding
    decoding_t0 = time.time()
    # Apply DeCout o DeCin's(message)
    # Running Time: O(n1*(n2*k2))   ~O(log**3(n))
    # Memory:       O(n1)           ~O(log(n))
    decoding = apply_decoding(k1, k2, n1, n2, d, r, cout, cins_list)
    decoding_t = time.time() - decoding_t0

    if is_ui:
        print("Message: ", message)
        print("Decoding: ", decoding)
        print(compare_matricies(message, decoding, ("Message", "Decoding")))

    return int(message == decoding), encoding_t, decoding_t


def run_test(n, d, occurrences):
    # Description: Run and log several occurrences of check_code_correctness(k1, d, k2) event

    # Initialize params
    k2 = math.ceil(math.log2(10*d*math.log2(n)))
    k1 = math.ceil(math.log2(n)/k2)
    #if(n < 100*(d**2)):
    #    raise ValueError("n < 100*d**2 has to hold, n: ", n, " 100*d**2: ", 100*(d**2))

    # Run check_code_correctness occurrences times
    success_occurrences = 0
    encoding_total_time = 0
    decoding_total_time = 0
    for i in range(occurrences):
        success, encoding_time, decoding_time = check_code_correctness(k1, d, k2, False)
        success_occurrences += success
        encoding_total_time += encoding_time
        decoding_total_time += decoding_time

    # Computes average encoding/decoding time
    encoding_average_time = encoding_total_time/occurrences
    decoding_average_time = decoding_total_time/occurrences

    return success_occurrences, encoding_average_time, decoding_average_time


def statistics(upper, lower, step, occurrences, d):
    # Description: Runs statistics on param n
    print('checking runtime n')

    successes = {}
    encoding_times = {}
    decoding_times = {}

    for n in range(lower, upper, step):
        #d = math.ceil(math.sqrt(n) / 10)
        # Run test function
        success_occurrences, encoding_average_time, decoding_average_time = run_test(n, d, occurrences)
        # Save data to hash table
        successes[n] = success_occurrences
        encoding_times[n] = encoding_average_time
        decoding_times[n] = decoding_average_time

        print(" n: ", n, " d: ", d, " Encoding runtime: ", encoding_average_time,
              " Decoding runtime: ", decoding_average_time, " Success occurrences: ", success_occurrences,
              " Success ratio: ", (success_occurrences/occurrences))

    print("=================")
    return successes, encoding_times, decoding_times


def build_figure(data, titles, param_name, d):
    # Description build plots
    # Input:
        # data [List[Dictionary[Int, Int]]]

    # Const
    MARGIN = 0.05

    # Figure settings
    fig, axes = plt.subplots(nrows=2, ncols=math.ceil(len(data)/2))
    plt.subplots_adjust(hspace=0.75, left=0.2, right=0.9)
    fig.set_size_inches(18.5, 10.5)

    # Figure data
    fig.text(MARGIN, 1 - MARGIN, 'd = ' + str(d), fontsize=18)
    # Build plots
    for ax, coordinates, title, iteration in zip(axes.flat[:], data, titles, range(len(data))):
        ax.set_title(title + " graph")
        ax.set_xlabel(param_name)
        ax.set_ylabel(title)

        # Data
        keys = sorted(coordinates.keys())
        values = sorted(coordinates.values())
        std = np.array(values).std()
        mean = np.array(values).mean()
        fig.text(MARGIN, 1 - (iteration+1)*4*MARGIN, title + "\n" + "std: " + str(round(std, 2)) + "\nmean: " + str(round(mean, 2)), fontsize=18)
        # Graph received by results
        ax.plot(keys, values, color='k', linestyle='dashed', marker='o')
        # Expected graph

    plt.show()


if __name__ == '__main__':
    tests = 100
    upper = 80001
    lower = 100
    step = math.ceil((upper - lower) / tests)
    occurrences = 10
    d = 3
    successes, encoding_times, decoding_times = statistics(upper, lower, step, occurrences, d)
    successes_ratio = {k: v / occurrences for k, v in successes.items()}
    data = [successes_ratio, encoding_times, decoding_times]
    build_figure(data, ["Success ratio", "Encoding time", "Decoding time"], "n", d)

