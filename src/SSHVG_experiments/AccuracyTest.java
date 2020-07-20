package SSHVG_experiments;

import java.io.File;
import java.io.FileWriter;
import java.io.Writer;

import utilities.ClassifierTools;
import weka.classifiers.meta.timeseriesensembles.WeightedEnsemble;
import weka.core.Instances;
import weka.core.shapelet.QualityMeasures;
import weka.filters.timeseries.shapelet_transforms.classValue.BinarisedClassValue;
import weka.filters.timeseries.shapelet_transforms.fss.ShapeletTransformWithSubclassSampleAndLFDP;
import weka.filters.timeseries.shapelet_transforms.sshvg.ShapeletTransformWithHVG;
import weka.filters.timeseries.shapelet_transforms.subsequenceDist.ImprovedOnlineSubSeqDistance;
import fileIO.DataSets;

public class AccuracyTest {
	public static void main(String[] args) throws Exception  {

		String[] problems = { "Adiac", // 390,391,176,37
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
		for (int i = 0; i < problems.length; i++) {
			System.out.print(problems[i] + "\t");
			accuracyForSSHVG(problems[i]);
			System.out.println();

		}
	}

	public static void accuracyForFSS(String problem) throws Exception {
		final String resampleLocation = DataSets.problemPath;
		final String dataset = problem;
		final String filePath = resampleLocation + File.separator + dataset
				+ File.separator + dataset;

		Instances test, train;
		test = utilities.ClassifierTools.loadData(filePath + "_TEST");
		train = utilities.ClassifierTools.loadData(filePath + "_TRAIN");
		ShapeletTransformWithSubclassSampleAndLFDP transform = new ShapeletTransformWithSubclassSampleAndLFDP();
		transform.setRoundRobin(true);

		transform.setClassValue(new BinarisedClassValue());
		transform.setSubSeqDistance(new ImprovedOnlineSubSeqDistance());
		transform.useCandidatePruning();
		transform.setNumberOfShapelets(train.numInstances() / 2);
		transform
				.setQualityMeasure(QualityMeasures.ShapeletQualityChoice.INFORMATION_GAIN);

		Instances tranTrain = transform.process(train);
		Instances tranTest = transform.process(test);

		double accuracy;
		WeightedEnsemble we = new WeightedEnsemble();
		we.buildClassifier(tranTrain);
		accuracy = ClassifierTools.accuracy(tranTest, we);
		System.out.print(accuracy + "\t");

	}

	public static void accuracyForSSHVG(String problem) throws Exception {

		final String resampleLocation = DataSets.problemPath;
		final String dataset = problem;
		final String filePath = resampleLocation + File.separator + dataset
				+ File.separator + dataset;
		Instances test, train;
		test = utilities.ClassifierTools.loadData(filePath + "_TEST");
		train = utilities.ClassifierTools.loadData(filePath + "_TRAIN");
		// use fold as the seed.
		// train = InstanceTools.subSample(train, 100, fold);
		ShapeletTransformWithHVG transform = new ShapeletTransformWithHVG();
		transform.setRoundRobin(true);
		// construct shapelet classifiers.
		transform.setClassValue(new BinarisedClassValue());
		transform.setSubSeqDistance(new ImprovedOnlineSubSeqDistance());
		transform.useCandidatePruning();
		transform.setNumberOfShapelets(train.numInstances() / 2);
		transform
				.setQualityMeasure(QualityMeasures.ShapeletQualityChoice.INFORMATION_GAIN);

		
		Instances tranTrain = transform.process(train);
		
		Instances tranTest = transform.process(test);

		WeightedEnsemble we = new WeightedEnsemble();
		we.buildClassifier(tranTrain);
		double accuracy = ClassifierTools.accuracy(tranTest, we);
		
		System.out.print(accuracy + "\t");
	}

}
