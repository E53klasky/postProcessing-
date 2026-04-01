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
            "                --scale       <float>\n"
            "\n"
            "  Number of --error-files must match number of variables.\n"
            "  Error variable inside each file is assumed to be {varname}_truncation_error.\n"
            "  --block-mode: for each rank's local block, take abs min or abs max of the\n"
            "                truncation error field and use that as the MGARD tolerance.\n"
            "  --scale:      multiply the block tolerance by this factor (default 1.0).\n"
            "\n"
            "Example:\n"
            "  mpirun -n 4 compress_grid 2d pp,ux,uy input.bp compressed.bp \\\n"
            "         --error-files pp_diff.bp ux_diff.bp uy_diff.bp \\\n"
            "         --block-mode max --scale 0.5 \n";
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
    std::string block_mode = "max";
    double scale = 1.0;
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
        else if (arg == "--scale")
        {
            if (i + 1 < argc)
                opts.scale = std::stod(argv[++i]);
        }
    }
    return opts;
}


double block_tolerance(const std::vector<double> &data,
                       const std::string &mode,
                       double scale = 1.0)
{
    double result = 0.0;
    if (mode == "max")
    {
        for (double v : data)
            result = std::max(result, std::abs(v));
    }
    else // min
    {
        result = std::numeric_limits<double>::max();
        for (double v : data)
        {
            double av = std::abs(v);
            if (av > 0.0) result = std::min(result, av);
        }
        if (result == std::numeric_limits<double>::max()) result = 0.0;
    }
    result *= scale;

  
    constexpr double TOL_FLOOR = 1e-6;
   
    if (result < TOL_FLOOR) result = TOL_FLOOR;
 

    return result;
}


double compute_tolerance_from_error_file(
    adios2::IO        &err_io,
    adios2::Engine    &err_reader,
    const std::string &err_vname,
    const std::vector<size_t> &squeezed_start,   // this rank's start in each spatial dim
    const std::vector<size_t> &squeezed_count,   // this rank's count in each spatial dim
    const std::string &mode,
    double             scale,
    int                rank)
{
    adios2::Variable<double> err_var = err_io.InquireVariable<double>(err_vname);
    if (!err_var)
    {
        if (rank == 0)
            std::cerr << "**** Warning: '" << err_vname
                      << "' not found in error file — using fallback 0.001. ****\n";
        return 0.001;
    }

    const auto   err_shape = err_var.Shape();
    const size_t err_ndims = err_shape.size();

    std::vector<size_t> full_start(err_ndims, 0);
    std::vector<size_t> full_count = err_shape;   

    size_t err_total = 1;
    for (auto c : full_count) err_total *= c;

    err_var.SetSelection({full_start, full_count});
    std::vector<double> err_full(err_total);
    err_reader.Get(err_var, err_full.data(), adios2::Mode::Sync);

    std::vector<size_t> err_spatial_dims; 
    std::vector<size_t> err_spatial_size;  
    for (size_t d = 0; d < err_ndims; ++d)
        if (err_shape[d] != 1)
        {
            err_spatial_dims.push_back(d);
            err_spatial_size.push_back(err_shape[d]);
        }

    const size_t nspatial = err_spatial_dims.size();

    if (squeezed_start.size() < nspatial)
    {
        if (rank == 0)
            std::cerr << "Warning: dim mismatch between error file and main variable"
                      << " — using fallback 0.001.\n";
        return 0.001;
    }

    std::vector<size_t> strides(nspatial, 1);
    for (int d = (int)nspatial - 2; d >= 0; --d)
        strides[d] = strides[d + 1] * err_spatial_size[d + 1];

    std::vector<double> err_block;
    err_block.reserve(1);  

    size_t block_total = 1;
    for (size_t d = 0; d < nspatial; ++d) block_total *= squeezed_count[d];
    err_block.resize(block_total);

    for (size_t flat = 0; flat < block_total; ++flat)
    {
        size_t tmp = flat;
        size_t full_flat = 0;
        for (int d = (int)nspatial - 1; d >= 0; --d)
        {
            size_t local_coord = tmp % squeezed_count[d];
            tmp /= squeezed_count[d];
            size_t global_coord = squeezed_start[d] + local_coord;
            full_flat += global_coord * strides[d];
        }
        err_block[flat] = err_full[full_flat];
    }

    double tol = block_tolerance(err_block, mode, scale);
    if (tol == 0.0) tol = 1e-15;
    return tol;
}


int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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
                  << "\nScale       : " << opts.scale
                  << "\nError files : ";
        if (use_error_files)
            for (auto &f : opts.error_files) std::cout << f << " ";
        else
            std::cout << "(none — using fixed tolerance 0.001)";
        std::cout << "\n\n";
    }

    std::vector<int> mpi_dims(ndims, 0);
    MPI_Dims_create(size, ndims, mpi_dims.data());
    std::vector<int> periods(ndims, 0);
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, ndims, mpi_dims.data(), periods.data(), 0, &cart_comm);

    std::vector<int> coords(ndims, 0);
    MPI_Cart_coords(cart_comm, rank, ndims, coords.data());

    adios2::ADIOS adios(cart_comm);

    adios2::IO reader_io = adios.DeclareIO("ReaderIO");
    adios2::Engine reader = reader_io.Open(input_file, adios2::Mode::Read);

    adios2::IO writer_io = adios.DeclareIO("WriterIO");
    adios2::Operator mgardOp = adios.DefineOperator("MGARDCompressor", "mgard");
    adios2::Engine writer = writer_io.Open(output_file, adios2::Mode::Write);

    adios2::ADIOS adios_serial(MPI_COMM_SELF);

    std::vector<adios2::IO>     err_ios;
    std::vector<adios2::Engine> err_readers;

    if (use_error_files)
    {
        for (size_t vi = 0; vi < vars.size(); ++vi)
        {
            std::string io_name = "ErrorReaderIO_" + std::to_string(vi);
            adios2::IO eio = adios_serial.DeclareIO(io_name);
            adios2::Engine erd = eio.Open(opts.error_files[vi], adios2::Mode::Read);
            err_ios.push_back(std::move(eio));
            err_readers.push_back(std::move(erd));
        }
    }

    std::vector<bool> var_defined(vars.size(), false);

    while (true)
    {
        if (reader.BeginStep() != adios2::StepStatus::OK) break;

        if (use_error_files)
        {
            for (auto &erd : err_readers)
            {
                if (erd.BeginStep() != adios2::StepStatus::OK)
                {
                    if (rank == 0)
                        std::cerr << "Error: error file ran out of steps before main file.\n";
                    goto done;
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
                tolerance = compute_tolerance_from_error_file(
                    err_ios[vi],
                    err_readers[vi],
                    err_vname,
                    squeezed_start,
                    squeezed_count,
                    opts.block_mode,
                    opts.scale,
                    rank);
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
