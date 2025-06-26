import sys
import argparse
import adios2 
import ReaderClass
import WrighterClass

# clean up and parallel 
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate streamline plots from ADIOS2 BP5 files"
    )

    parser.add_argument(
        "--bpfile1",
        "--bp1",
        type=str,
        required=True,
        help="First Adios file with streamline segments (lower resolution/compressed)",
    )
    
    parser.add_argument("--readIO", "-rio", type=str, default="reader1", help="IO Name for the first Adios file (default: reader1)")
    
    parser.add_argument("--WrightIO", "-wio", type=str, default="writer1", help="IO Name for the second Adios file (default: writer1)")

    parser.add_argument("--error_bound", "-eb", type=float, required=True, default=None, help="Error bound for compression in xml file (required)")
    parser.add_argument(
        "--xml",
        "-x",
        type=str,
        default=None,
        help="Path to ADIOS2 XML configuration file (optional)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="compressed.bp",
        help="Output file name (default: compressed.bp)",
    )

    parser.add_argument(
        "--compres_step",
        "-c",
        type=int,
        default=None,
        help="If provided, compress only this specific step (optional)",
    )

    return parser.parse_args()



def main():
    parser = parse_arguments()
    bpfile1 = parser.bpfile1
    readIO = parser.readIO
    WrightIO = parser.WrightIO
    xml = parser.xml
    output = parser.output
    compres_step = parser.compres_step
    r = ReaderClass.Reader(readIO, bpfile1, xml=xml)
    w = WrighterClass.Writer(WrightIO, output, xml=xml)
    if xml is None:
        print("No XML file provided ending program.")
        sys.exit(1)
    flag = True
    while True:
        status = r.begin_step()
        if status != adios2.bindings.StepStatus.OK:
            break
        w.begin_step()
        
        if compres_step is None or r.current_step == compres_step:
            for name, info in r.Adios_reader.available_variables().items():
                r.set_read_vars([name])
                data = r.read_step(name)
                w.write(name, data)
                if flag:
                    w.write("error_bound", parser.error_bound)
                    flag = False
            
        r.end_step()
        w.end_step()
    r.close()
    w.close()
    print(f"Compression completed. Output written to {output}.")

        

if __name__ == "__main__":
    main()
