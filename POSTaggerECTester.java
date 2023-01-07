import java.util.HashMap;

/**
 * Testing class for Part of Speech (POS) Tagger, POSTaggerEC.java class
 *
 * @author Andy Xu & Kabir Moghe, Dartmouth CS 10, Fall 2022
 */

public class POSTaggerECTester{
    public static void main(String[] args){
        POSTaggerEC pt = new POSTaggerEC();

        // tests with hard-coded graphs

        //First hard-coded graph from Programming Drill
        // hard-coded tag probability table
        HashMap<String, HashMap<String, Double>> tagProb = new HashMap<String, HashMap<String, Double>>();

        HashMap<String, Double> row1 = new HashMap<String, Double>();
        row1.put("NP", -1.253);
        row1.put("N", -0.336);
        HashMap<String, Double> row2 = new HashMap<String, Double>();
        row2.put("V", 0.0);
        HashMap<String, Double> row3 = new HashMap<String, Double>();
        row3.put("NP", -1.504);
        row3.put("N", -0.405);
        row3.put("CNJ", -2.197);
        HashMap<String, Double> row4 = new HashMap<String, Double>();
        row4.put("V", -0.288);
        row4.put("CNJ", -1.386);
        HashMap<String, Double> row5 = new HashMap<String, Double>();
        row5.put("NP", -1.099);
        row5.put("N", -1.099);
        row5.put("V", -1.099);

        tagProb.put("#", row1);
        tagProb.put("NP", row2);
        tagProb.put("V", row3);
        tagProb.put("N", row4);
        tagProb.put("CNJ", row5);

        //hard-coded observation probability table
        HashMap<String, HashMap<String, Double>> obProb = new HashMap<String, HashMap<String, Double>>();

        HashMap<String, Double> obRow1 = new HashMap<String, Double>();
        obRow1.put("chase", 0.0);
        HashMap<String, Double> obRow2 = new HashMap<String, Double>();
        obRow2.put("chase", -1.504);
        obRow2.put("watch", -0.405);
        obRow2.put("get", -2.197);
        HashMap<String, Double> obRow3 = new HashMap<String, Double>();
        obRow3.put("watch", -1.792);
        obRow3.put("cat", -0.875);
        obRow3.put("dog", -0.875);
        HashMap<String, Double> obRow4 = new HashMap<String, Double>();
        obRow4.put("and", 0.0);

        obProb.put("NP", obRow1);
        obProb.put("V", obRow2);
        obProb.put("N", obRow3);
        obProb.put("CNJ", obRow4);

        pt.setProb(tagProb, obProb);
        System.out.println("Testing Viterbi decoding based on hard-coded graph from programming drill:");
        System.out.println("Sentence: cat chase dog | Predicted Tags: " + pt.viterbi("cat chase dog"));
        System.out.println("Sentence: dog chase cat | Predicted Tags: " + pt.viterbi("dog chase cat"));
        System.out.println("Sentence: chase watch dog chase watch | Predicted Tags: " + pt.viterbi("chase watch dog chase watch"));
        System.out.println("Sentence: dog watch cat chase dog | Predicted Tags: " + pt.viterbi("dog watch cat chase dog"));

        //Second Hard-Coded Graph
        //hard-coded tag probability table
        tagProb = new HashMap<String, HashMap<String, Double>>();
        row1 = new HashMap<String, Double>();
        row1.put("N", 0.0);
        row2 = new HashMap<String, Double>();
        row2.put("ADV", -0.405);
        row2.put("N",-1.099);
        row3 = new HashMap<String, Double>();
        row3.put("V", 0.0);

        tagProb.put("#", row1);
        tagProb.put("V", row2);
        tagProb.put("N", row3);
        tagProb.put("ADV", new HashMap<String, Double>());

        //hard-coded observation probability table
        obProb = new HashMap<String, HashMap<String, Double>>();
        obRow1 = new HashMap<String, Double>();
        obRow1.put("quickly", 0.0);
        obRow2 = new HashMap<String, Double>();
        obRow2.put("run", 0.0);
        obRow3 = new HashMap<String, Double>();
        obRow3.put("cat", -0.405);
        obRow3.put("dog", -1.099);

        obProb.put("ADV", obRow1);
        obProb.put("V", obRow2);
        obProb.put("N", obRow3);

        pt.setProb(tagProb, obProb);
        System.out.println("\nTesting Viterbi decoding based on our own hard-coded graph:");
        System.out.println("Sentence: cat run dog | Predicted Tags: " + pt.viterbi("cat run dog"));
        System.out.println("Sentence: dog run cat | Predicted Tags: " + pt.viterbi("dog run cat"));
        System.out.println("Sentence: dog run quickly | Predicted Tags: " + pt.viterbi("dog run quickly"));
        System.out.println("Sentence: cat run man | Predicted Tags: " + pt.viterbi("cat run man"));

        // Tests with simple files (have to set new txt files and train with new data first)
        String trainSentPathName = "ps5/texts/simple-train-sentences.txt";
        String trainTagPathName = "ps5/texts/simple-train-tags.txt";

        pt.setTrainingData(trainSentPathName, trainTagPathName);
        pt.train();

        String testSentPathName = "ps5/texts/simple-test-sentences.txt";
        String testTagPathName = "ps5/texts/simple-test-tags.txt";

        pt.testTagger(testSentPathName, testTagPathName);

        // Tests with Brown files
        trainSentPathName = "ps5/texts/brown-train-sentences.txt";
        trainTagPathName = "ps5/texts/brown-train-tags.txt";

        pt.setTrainingData(trainSentPathName, trainTagPathName);
        pt.train();

        testSentPathName = "ps5/texts/brown-test-sentences.txt";
        testTagPathName = "ps5/texts/brown-test-tags.txt";

        pt.testTagger(testSentPathName, testTagPathName);

        // Tests with our own example sentences
        System.out.println("\nTesting our own example sentences");
        System.out.println("Sentence: my favorite food is pizza . | Predicted Tags: " + pt.viterbi("my favorite food is pizza . "));
        System.out.println("Sentence: eat eat eat eat eat eat . | Predicted Tags: " + pt.viterbi("eat eat eat eat eat eat ."));
        System.out.println("Sentence: superman flew into the sun holding kryptonite . | Predicted Tags: " + pt.viterbi("superman flew into the sun holding kryptonite . "));

        // Tests tagging individual input line
        pt.tagLine();

        // Tests predicting next word given individual input line
        pt.bestNextWord();
    }
}
