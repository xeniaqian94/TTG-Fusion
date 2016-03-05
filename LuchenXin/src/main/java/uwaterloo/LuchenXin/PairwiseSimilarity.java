package uwaterloo.LuchenXin;

import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.kohsuke.args4j.ParserProperties;


/**
 * Hello world!
 *
 */
public class PairwiseSimilarity 
{
	public static class Args {

		@Option(name = "-runtag1", metaVar = "[path]", required = true, usage = "run path")
		public String runtag1;
		@Option(name = "-runtag2", metaVar = "[path]", required = true, usage = "run path")
		public String runtag2;
		@Option(name = "-topic", metaVar = "[path]", required = true, usage = "run path")
		public String topic;
	}
    public static void main( String[] argv )
    {
    	Args args = new Args();
		CmdLineParser parser = new CmdLineParser(args, ParserProperties.defaults().withUsageWidth(100));

		try {
			parser.parseArgument(argv);
		} catch (CmdLineException e) {
			System.err.println(e.getMessage());
			parser.printUsage(System.err);
			System.exit(-1);
		}
        System.out.println( args.runtag1+" "+args.runtag2+" "+args.topic );
    }
}
