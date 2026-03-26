#include <adios2.h>
#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <numeric>

void print_usage(int rank)
{
    if (rank == 0)
    {
        std::cerr <<
            "Usage:\n"
            "  compress_grid <2d|3d> <var1,var2,...> <input.bp> <output.bp>\n"
            "                --error-files <err1.bp> <err2.bp> ...\n"
            "                --block-mode  <min|max>\n"
            "\n"
            "  Number of --error-files must match number of variables.\n"
            "  Error variable inside each file is assumed to be {varname}_truncation_error.\n"
            "  --block-mode: for each rank's local block, take abs min or abs max of the\n"
            "                truncation error field and use that as the MGARD tolerance.\n"
            "\n"
            "Example:\n"
            "  mpirun -n 4 compress_grid 2d pp,ux,uy input.bp compressed.bp \\\n"
            "         --error-files pp_diff.bp ux_diff.bp uy_diff.bp \\\n"
            "         --block-mode max\n";
    }
}

std::vector<std::string> split_csv(const std::string &s)
{
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) out.push_back(tok);
    return out;
}

struct Options
{
    std::vector<std::string> error_files;
    std::string block_mode = "max";   // "min" | "max"  (alpha extension: "min*alpha" etc.)
};

Options parse_flags(int argc, char **argv, int rank)
{
    Options opts;
    for (int i = 1; i < argc; ++i)
    {
        std::string arg(argv[i]);
        if (arg == "--error-files")
        {
            ++i;
            // Consume all following tokens that are not flags (don't start with '-')
            while (i < argc && argv[i][0] != '-')
            {
                opts.error_files.emplace_back(argv[i]);
                ++i;
            }
            --i;
        }
        else if (arg == "--block-mode")
        {
            if (i + 1 < argc)
            {
                opts.block_mode = argv[++i];
                if (opts.block_mode != "min" && opts.block_mode != "max")
                {
                    if (rank == 0)
                        std::cerr << "Warning: unknown --block-mode '" << opts.block_mode
                                  << "', defaulting to 'max'\n";
                    opts.block_mode = "max";
                }
            }
        }
        // Skip positional args (indices 1-4) and anything else silently
    }
    return opts;
}

// Compute abs-min or abs-max of a data vector
double block_tolerance(const std::vector<double> &data, const std::string &mode)
{
    double result = 0.0;
    if (mode == "max")
    {
        for (double v : data)
            result = std::max(result, std::abs(v));
    }
    else // min — skip exact zeros to avoid a tolerance of 0
    {
        result = std::numeric_limits<double>::max();
        for (double v : data)
        {
            double av = std::abs(v);
            if (av > 0.0) result = std::min(result, av);
        }
        if (result == std::numeric_limits<double>::max()) result = 0.0;
    }
    return result;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Positional args: <2d|3d> <vars> <input.bp> <output.bp>
    const int N_POS = 4;
    if (argc < N_POS + 1)
    {
        print_usage(rank);
        MPI_Finalize();
        return -1;
    }

    std::string dim_type(argv[1]);
    std::vector<std::string> vars = split_csv(argv[2]);
    std::string input_file(argv[3]);
    std::string output_file(argv[4]);

    Options opts = parse_flags(argc, argv, rank);

    // Validate error files count
    if (!opts.error_files.empty() && opts.error_files.size() != vars.size())
    {
        if (rank == 0)
            std::cerr << "Error: --error-files count (" << opts.error_files.size()
                      << ") must match variable count (" << vars.size() << ").\n";
        MPI_Finalize();
        return -1;
    }

    const bool use_error_files = !opts.error_files.empty();
    const int ndims = (dim_type == "2d") ? 2 : 3;

    if (rank == 0)
    {
        std::cout << "Variables   : ";
        for (auto &v : vars) std::cout << v << " ";
        std::cout << "\nInput       : " << input_file
                  << "\nOutput      : " << output_file
                  << "\nBlock-mode  : " << opts.block_mode
                  << "\nError files : ";
        if (use_error_files)
            for (auto &f : opts.error_files) std::cout << f << " ";
        else
            std::cout << "(none — using fixed tolerance 0.001)";
        std::cout << "\n\n";
    }

    // -------------------------------------------------------------------
    // MPI Cartesian topology
    // -------------------------------------------------------------------
    std::vector<int> mpi_dims(ndims, 0);
    MPI_Dims_create(size, ndims, mpi_dims.data());
    std::vector<int> periods(ndims, 0);
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, ndims, mpi_dims.data(), periods.data(), 0, &cart_comm);

    std::vector<int> coords(ndims, 0);
    MPI_Cart_coords(cart_comm, rank, ndims, coords.data());

    // -------------------------------------------------------------------
    // ADIOS2 setup — main data
    // -------------------------------------------------------------------
    adios2::ADIOS adios(cart_comm);

    adios2::IO reader_io = adios.DeclareIO("ReaderIO");
    adios2::Engine reader = reader_io.Open(input_file, adios2::Mode::Read);

    adios2::IO writer_io = adios.DeclareIO("WriterIO");
    adios2::Operator mgardOp = adios.DefineOperator("MGARDCompressor", "mgard");
    adios2::Engine writer = writer_io.Open(output_file, adios2::Mode::Write);

    // -------------------------------------------------------------------
    // ADIOS2 setup — one reader per error file
    // -------------------------------------------------------------------
    std::vector<adios2::IO>     err_ios;
    std::vector<adios2::Engine> err_readers;

    if (use_error_files)
    {
        for (size_t vi = 0; vi < vars.size(); ++vi)
        {
            std::string io_name = "ErrorReaderIO_" + std::to_string(vi);
            adios2::IO eio = adios.DeclareIO(io_name);
            adios2::Engine erd = eio.Open(opts.error_files[vi], adios2::Mode::Read);
            err_ios.push_back(std::move(eio));
            err_readers.push_back(std::move(erd));
        }
    }

    std::vector<bool> var_defined(vars.size(), false);

    while (true)
    {
        if (reader.BeginStep() != adios2::StepStatus::OK) break;

        // Advance all error readers in lockstep
        if (use_error_files)
        {
            for (auto &erd : err_readers)
            {
                if (erd.BeginStep() != adios2::StepStatus::OK)
                {
                    if (rank == 0)
                        std::cerr << "Error: error file ran out of steps before main file.\n";
                    goto done;   // break out of outer while
                }
            }
        }

        writer.BeginStep();

        for (size_t vi = 0; vi < vars.size(); ++vi)
        {
            const std::string &vname = vars[vi];

            adios2::Variable<double> var = reader_io.InquireVariable<double>(vname);
            if (!var)
            {
                if (rank == 0) std::cerr << "Variable '" << vname << "' not found — skipping.\n";
                continue;
            }

            // Raw shape from file (may contain leading/trailing size-1 dims)
            const auto raw_shape   = var.Shape();
            const size_t ndims_var = raw_shape.size();

            std::vector<size_t> start(ndims_var, 0);
            std::vector<size_t> count(ndims_var, 0);

            int spatial_idx = 0;
            for (size_t i = 0; i < ndims_var; ++i)
            {
                if (raw_shape[i] == 1)
                {
                    start[i] = 0;
                    count[i] = 1;
                }
                else if (spatial_idx < ndims)
                {
                    size_t block  = raw_shape[i] / mpi_dims[spatial_idx];
                    start[i]      = coords[spatial_idx] * block;
                    count[i]      = (coords[spatial_idx] == mpi_dims[spatial_idx] - 1)
                                        ? raw_shape[i] - start[i]
                                        : block;
                    ++spatial_idx;
                }
                else
                {
                    start[i] = 0;
                    count[i] = raw_shape[i];
                }
            }

            std::vector<size_t> squeezed_shape, squeezed_start, squeezed_count;
            for (size_t i = 0; i < ndims_var; ++i)
            {
                if (raw_shape[i] != 1)
                {
                    squeezed_shape.push_back(raw_shape[i]);
                    squeezed_start.push_back(start[i]);
                    squeezed_count.push_back(count[i]);
                }
            }

            size_t total = 1;
            for (auto c : count) total *= c;

            var.SetSelection({start, count});
            std::vector<double> local_data(total);
            reader.Get(var, local_data.data(), adios2::Mode::Sync);

            double tolerance = 0.001;

            if (use_error_files)
            {
                const std::string err_vname = vname + "_truncation_error";
                adios2::Variable<double> err_var =
                    err_ios[vi].InquireVariable<double>(err_vname);

                if (!err_var)
                {
                    if (rank == 0)
                        std::cerr << "Warning: '" << err_vname
                                  << "' not found in error file — using fallback 0.001.\n";
                }
                else
                {
                    const auto err_shape    = err_var.Shape();
                    const size_t err_ndims  = err_shape.size();

                    std::vector<size_t> err_spatial;
                    for (size_t d = 0; d < err_ndims; ++d)
                        if (err_shape[d] != 1) err_spatial.push_back(d);

                    std::vector<size_t> err_start(err_ndims, 0);
                    std::vector<size_t> err_count(err_ndims, 0);
                    for (size_t d = 0; d < err_ndims; ++d)
                    {
                        if (err_shape[d] == 1)
                        {
                            err_start[d] = 0;
                            err_count[d] = 1;
                        }
                        else
                        {
                            size_t sidx = 0;
                            for (size_t dd = 0; dd < d; ++dd)
                                if (err_shape[dd] != 1) ++sidx;

                            if (sidx < squeezed_start.size())
                            {
                                err_start[d] = squeezed_start[sidx];
                                err_count[d] = squeezed_count[sidx];
                            }
                            else
                            {
                                err_start[d] = 0;
                                err_count[d] = err_shape[d];
                            }
                        }
                    }

                    size_t err_total = 1;
                    for (auto c : err_count) err_total *= c;

                    err_var.SetSelection({err_start, err_count});
                    std::vector<double> err_data(err_total);
                    err_readers[vi].Get(err_var, err_data.data(), adios2::Mode::Sync);

                    tolerance = block_tolerance(err_data, opts.block_mode);

                    if (tolerance == 0.0) tolerance = 1e-15;
                }
            }

            adios2::Variable<double> wvar;
            if (!var_defined[vi])
            {
                wvar = writer_io.DefineVariable<double>(
                    vname, squeezed_shape, squeezed_start, squeezed_count);
                wvar.AddOperation(mgardOp, {{"accuracy", std::to_string(tolerance)},
                                            {"mode", "REL"}});
                var_defined[vi] = true;
            }
            else
            {
                wvar = writer_io.InquireVariable<double>(vname);
                wvar.SetSelection({squeezed_start, squeezed_count});
                wvar.RemoveOperations();
                wvar.AddOperation(mgardOp, {{"accuracy", std::to_string(tolerance)},
                                            {"mode", "REL"}});
            }

            {
                std::ostringstream ss;
                ss << "[Rank " << rank << "] " << vname << " block=(";
                for (size_t i = 0; i < squeezed_count.size(); ++i)
                    ss << squeezed_count[i] << (i + 1 < squeezed_count.size() ? "x" : "");
                ss << ")  tolerance=" << tolerance << "\n";
                std::cout << ss.str();
            }

            writer.Put(wvar, local_data.data(), adios2::Mode::Sync);
        }

        writer.EndStep();
        reader.EndStep();

        if (use_error_files)
            for (auto &erd : err_readers)
                erd.EndStep();
    }

done:
    reader.Close();
    writer.Close();
    if (use_error_files)
        for (auto &erd : err_readers)
            erd.Close();

    if (rank == 0) std::cout << "Done. Output: " << output_file << "\n";

    MPI_Finalize();
    return 0;
}
