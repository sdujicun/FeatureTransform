package SSHVG_experiments;

import java.io.File;
import java.io.FileWriter;
import java.io.Writer;

import utilities.ClassifierTools;
import weka.classifiers.meta.timeseriesensembles.WeightedEnsemble;
import weka.core.Instances;
import weka.core.shapelet.QualityMeasures;
import weka.filters.timeseries.shapelet_transforms.classValue.BinarisedClassValue;
import weka.filters.timeseries.shapelet_transforms.sshvg.ShapeletTransformWithHVG;
import weka.filters.timeseries.shapelet_transforms.subsequenceDist.ImprovedOnlineSubSeqDistance;
import fileIO.DataSets;

public class ShapeletNumberExperiment {
	public static void main(String[] args) throws Exception {

		
		String[] problems = { 
				"Adiac", // 390,391,176,37
				"Beef", // 30,30,470,5
				"ChlorineConcentration", 
				"Coffee", // 28,28,286,2
				"DiatomSizeReduction", // 16,306,345,4
				"ItalyPowerDemand", // 67,1029,24,2
				"Lightning7", // 70,73,319,7
				"MedicalImages", // 381,760,99,10
				"MoteStrain", // 20,1252,84,2
				"Symbols", // 25,995,398,6
				"Trace", // 100,100,275,4
				"TwoLeadECG", // 23,1139,82,2
		};
		System.out.println(problems.length);
		double[] rate={0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0};
		for (int i = 0; i < problems.length; i++) {
			for(int j=0;j<rate.length;j++){
				shapeletNumberExperiments(problems[i],rate[j]);
			}
		}

	}

	public static void shapeletNumberExperiments(String problem,double rate)
			throws Exception {

		final String resampleLocation = DataSets.problemPath;
		final String dataset = problem;
		final String filePath = resampleLocation + File.separator + dataset
				+ File.separator + dataset;

		Instances test, train;
		test = utilities.ClassifierTools.loadData(filePath + "_TEST");
		train = utilities.ClassifierTools.loadData(filePath + "_TRAIN");

		ShapeletTransformWithHVG transform = new ShapeletTransformWithHVG();
		transform.setRoundRobin(true);

		transform.setClassValue(new BinarisedClassValue());
		transform.setSubSeqDistance(new ImprovedOnlineSubSeqDistance());
		transform.useCandidatePruning();
		transform.setNumberOfShapelets((int)(train.numInstances()*rate));
		transform
				.setQualityMeasure(QualityMeasures.ShapeletQualityChoice.INFORMATION_GAIN);

		long d1 = System.nanoTime();
		Instances tranTrain = transform.process(train);
		long d2 = System.nanoTime();
		double time = (d2 - d1) * 0.000000001;

//		Instances tranTest = transform.process(test);
//
//		double accuracy;
//		WeightedEnsemble we = new WeightedEnsemble();
//		we.buildClassifier(tranTrain);
//		accuracy = ClassifierTools.accuracy(tranTest, we);

		System.out.println(problem + "\t" + rate + "\t" + time);
		

		//System.out.println(problem + "\t" + rate + "\t" + time + "\t" + accuracy);

	}

}
