#include <adios2.h>
#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>

void print_usage(int rank)
{
    if(rank == 0)
    {
        std::cerr << "Usage: compress_grid <2d|3d> <var1,var2,...> <input.bp> <output.bp>\n";
        std::cerr << "Example: compress_grid 2d pp,ux,uy input.bp compressed.bp\n";
    }
}

std::vector<std::string> split_vars(const std::string &var_str)
{
    std::vector<std::string> vars;
    std::stringstream ss(var_str);
    std::string v;
    while (std::getline(ss, v, ',')) vars.push_back(v);
    return vars;
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(argc < 5)
    {
        print_usage(rank);
        MPI_Finalize();
        return -1;
    }

    std::string dim_type(argv[1]);
    std::vector<std::string> vars = split_vars(argv[2]);
    std::string input_file(argv[3]);
    std::string output_file(argv[4]);

    int ndims = (dim_type == "2d") ? 2 : 3;

    std::vector<int> mpi_dims(ndims, 0);
    MPI_Dims_create(size, ndims, mpi_dims.data());
    std::vector<int> periods(ndims, 0);
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, ndims, mpi_dims.data(), periods.data(), 0, &cart_comm);

    int coords[3] = {0,0,0};
    MPI_Cart_coords(cart_comm, rank, ndims, coords);

    adios2::ADIOS adios(cart_comm);
    adios2::IO reader_io = adios.DeclareIO("ReaderIO");
    adios2::Engine reader = reader_io.Open(input_file, adios2::Mode::Read);

    adios2::IO writer_io = adios.DeclareIO("WriterIO");
    

    // adios2::Operator zfpOp = adios.DefineOperator("ZFPCompressor", "zfp");
    // writer_io.AddOperation(zfpOp, {{"accuracy", "0.0001"}});

    adios2::Engine writer = writer_io.Open(output_file, adios2::Mode::Write);

    while(true)
    {
        const auto status = reader.BeginStep();
        if(status != adios2::StepStatus::OK) break;

        writer.BeginStep();

        for(const auto &vname : vars)
        {
            adios2::Variable<double> var = reader_io.InquireVariable<double>(vname);
            if(!var)
            {
                if(rank==0) std::cerr << "Variable " << vname << " not found.\n";
                continue;
            }

            const auto shape = var.Shape();
            const size_t actual_dims = shape.size();

            std::vector<size_t> start(actual_dims, 0);
            std::vector<size_t> count(actual_dims);

            int spatial_idx = 0;
            for(int i = 0; i < actual_dims; i++) {
                if (shape[i] > 1 && spatial_idx < ndims) {
                    size_t block = shape[i] / mpi_dims[spatial_idx];
                    start[i] = coords[spatial_idx] * block;
                    count[i] = (coords[spatial_idx] == mpi_dims[spatial_idx] - 1) ? shape[i] - start[i] : block;
                    spatial_idx++;
                } else {
                    start[i] = 0;
                    count[i] = shape[i];
                }
            }

            var.SetSelection({start, count});
            std::stringstream ss;
            ss << "[Rank " << rank << "] " << vname << " local block size: (";
            for(size_t i = 1; i < count.size(); ++i) {
                ss << count[i] << (i == count.size() - 1 ? "" : " x ");
            }
            ss << ")\n";
            std::cout << ss.str();

            size_t total = 1;
            for(auto c : count) total *= c;
            std::vector<double> local_data(total);

            reader.Get(var, local_data.data(), adios2::Mode::Sync);

            auto wvar = writer_io.InquireVariable<double>(vname);
            if (!wvar) {
                wvar = writer_io.DefineVariable<double>(vname, shape, start, count);
            } else {
                wvar.SetSelection({start, count});
            }

            writer.Put(wvar, local_data.data(), adios2::Mode::Sync);
        }

        writer.EndStep();
        reader.EndStep();
    }

    reader.Close();
    writer.Close();

    if(rank==0) std::cout << "Processing complete. Output: " << output_file << "\n";

    MPI_Finalize();
    return 0;
}
