import java.io.*;
import java.util.*;

/**
 * Part of Speech (POS) Tagger using a Hidden Markov Model and the Viterbi algorithm with an extra method to predict the best next word given a phrase/sentence
 *
 * @author Andy Xu & Kabir Moghe, Dartmouth CS 10, Fall 2022
 */

public class POSTaggerEC {
    private static String trainSentPathName = "ps5/texts/brown-train-sentences.txt"; //name of file containing sentences for training
    private static String trainTagPathName = "ps5/texts/brown-train-tags.txt"; //name of file containing tags for training

    private HashMap<String, HashMap<String, Integer>> tagData; //tag transitions data
    private HashMap<String, HashMap<String, Double>> tagProb; //tag transitions probability

    private HashMap<String, HashMap<String, Integer>> obsData; //observations data
    private HashMap<String, HashMap<String, Double>> obsProb; //observations probability

    private final double UNOBSERVED = -100.0; //unseen word penalty

    public POSTaggerEC(){
        tagData = new HashMap<String, HashMap<String, Integer>>();
        tagProb = new HashMap<String, HashMap<String, Double>>();
        obsData = new HashMap<String, HashMap<String, Integer>>();
        obsProb = new HashMap<String, HashMap<String, Double>>();
    }

    /**
     * Method to train: Creates a Hidden Markov Model (HMM) by determining the probability that any tag will transition
     * to another tag and the probability that some particular word corresponds to some tag
     */
    public void train(){
        tagData.put("#", new HashMap<String, Integer>());
        tagData.get("#").put("Normalize by", 0);

        //open files containing training data and tags
        BufferedReader sentInput = null, tagInput = null;
        try {
            sentInput = new BufferedReader(new FileReader(trainSentPathName));
            tagInput = new BufferedReader(new FileReader(trainTagPathName));
        }
        catch (FileNotFoundException e) {
            System.err.println("Cannot open file.\n" + e.getMessage());
        }

        //read through input files
        try{
            String sentNext = sentInput.readLine();
            String tagNext = tagInput.readLine();
            while(sentNext!=null && tagNext!=null){
                String[] sent = sentNext.split("\\s");
                String[] tag = tagNext.split("\\s");

                //iterate through current sentence
                for(int ind=0; ind<sent.length; ind++){
                    String curWord = sent[ind]; //current word
                    String curTag = tag[ind]; //tag for current word

                    //fill in tagData map
                    if(!tagData.containsKey(curTag)){
                        tagData.put(curTag, new HashMap<String, Integer>());
                        tagData.get(curTag).put("Normalize by", 0);
                    }

                    if(ind==0){
                        if(tagData.get("#").containsKey(curTag)) {
                            tagData.get("#").put(curTag, tagData.get("#").get(curTag)+1);
                        }
                        else{
                            tagData.get("#").put(curTag, 1);
                        }

                        tagData.get("#").put("Normalize by", tagData.get("#").get("Normalize by")+1);
                    }
                    else{
                        if(tagData.get(tag[ind-1]).containsKey(curTag)) {
                            tagData.get(tag[ind-1]).put(curTag, tagData.get(tag[ind-1]).get(curTag)+1);
                        }
                        else{
                            tagData.get(tag[ind-1]).put(curTag, 1);
                        }

                        tagData.get(tag[ind-1]).put("Normalize by", tagData.get(tag[ind-1]).get("Normalize by")+1);
                    }

                    //fill in obsData map
                    if(!obsData.containsKey(curTag)){
                        obsData.put(curTag, new HashMap<String, Integer>());
                        obsData.get(curTag).put("Normalize by", 0);
                    }

                    if(obsData.get(curTag).containsKey(curWord)){
                        obsData.get(curTag).put(curWord, obsData.get(curTag).get(curWord)+1);
                    }
                    else{
                        obsData.get(curTag).put(curWord, 1);
                    }
                    obsData.get(curTag).put("Normalize by", obsData.get(curTag).get("Normalize by")+1);

                }

                //read next line
                sentNext = sentInput.readLine();
                tagNext = tagInput.readLine();
            }
        }
        catch (IOException e) {
            System.err.println("IO error while reading.\n" + e.getMessage());
        }

        //Calculate tag transition probabilities
        for (HashMap.Entry<String,HashMap<String, Integer>> curTag1 : tagData.entrySet()) {
            if(!curTag1.getKey().equals("Normalize by")) {
                tagProb.put(curTag1.getKey(), new HashMap<String, Double>());

                for (HashMap.Entry<String, Integer> curTag2 : curTag1.getValue().entrySet()) {
                    if(!curTag2.getKey().equals("Normalize by")) {
                        double curProb = Math.log((double) curTag2.getValue() / curTag1.getValue().get("Normalize by"));
                        tagProb.get(curTag1.getKey()).put(curTag2.getKey(), curProb);
                    }
                }
            }
        }

        //Calculate observation probabilities
        for (HashMap.Entry<String,HashMap<String, Integer>> curObs1 : obsData.entrySet()) {
            if(!curObs1.getKey().equals("Normalize by")) {
                obsProb.put(curObs1.getKey(), new HashMap<String, Double>());

                for (HashMap.Entry<String, Integer> curObs2 : curObs1.getValue().entrySet()) {
                    if(!curObs2.getKey().equals("Normalize by")) {
                        double curProb = Math.log((double) curObs2.getValue() / curObs1.getValue().get("Normalize by"));
                        obsProb.get(curObs1.getKey()).put(curObs2.getKey(), curProb);
                    }
                }
            }
        }

        //close input files
        try {
            sentInput.close();
            tagInput.close();
        }
        catch (IOException e) {
            System.err.println("Cannot close file.\n" + e.getMessage());
        }
    }

    /**
     * Method that performs Viterbi decoding to identify the best sequence of tags for a given line
     *
     * @param line String of words
     */
    public ArrayList<String> viterbi(String line){
        ArrayList<String> sent = new ArrayList<String>(Arrays.asList(line.split("\\s"))); //ArrayList of words from sentence
        ArrayList<String> labels = new ArrayList<String>(); //ArrayList to store tags
        //ArrayList of maps to keep track of predecessor of each state
        ArrayList<HashMap<String, String>> pred = new ArrayList<HashMap<String, String>>();

        //starts observation sequence with "#" and score of 0.0
        sent.add(0, "#");
        ArrayList<String> currStates = new ArrayList<String>();
        currStates.add("#");
        HashMap<String, Double> currScores = new HashMap<String, Double>();
        currScores.put("#", 0.0);

        //iterate through words in sentence (observations)
        for(int ind=0; ind<sent.size()-1; ind++){
            ArrayList<String> nextStates = new ArrayList<String>();
            HashMap<String, Double> nextScores = new HashMap<String, Double>();
            pred.add(new HashMap<String, String>());

            //iterate through current states
            for(int curInd=0; curInd<currStates.size(); curInd++){
                //add possible next states to nextStates list
                for(HashMap.Entry<String, Double> trans : tagProb.get(currStates.get(curInd)).entrySet()){
                    if(!nextStates.contains(trans.getKey()))
                        nextStates.add(trans.getKey());

                    //calculate observed score
                    double obsScore = 0.0;
                    if(obsProb.get(trans.getKey()).containsKey(sent.get(ind+1)))
                        obsScore = obsProb.get(trans.getKey()).get(sent.get(ind+1));
                    else
                        obsScore = UNOBSERVED;

                    //calculate nextScore by adding current score, transition score, and observation score
                    double nextScore = currScores.get(currStates.get(curInd)) + trans.getValue() + obsScore;

                    //add next state to nextScores map if it is not already in it, or if the new nextScore is greater
                    //than the existing nextScore for the next state
                    if(!nextScores.containsKey(trans.getKey()) || nextScore > nextScores.get(trans.getKey())){
                        nextScores.put(trans.getKey(), nextScore);

                        //keep track of predecessor of next state
                        pred.get(ind).put(trans.getKey(), currStates.get(curInd));
                    }
                }
            }
            //set currStates and currScores to nextStates and nextScores for the next observation
            currStates = nextStates;
            currScores = nextScores;
        }

        //backtrack to identify path
        double maxScore = Double.MAX_VALUE*(-1.0);
        String maxKey = "";
        for(HashMap.Entry<String, Double> score : currScores.entrySet()){
            if(score.getValue()>maxScore){
                maxScore = score.getValue();
                maxKey = score.getKey();
            }
        }

        String next = maxKey;
        labels.add(maxKey);
        for(int ind=pred.size()-1; ind>0; ind--){
            labels.add(0, pred.get(ind).get(next));
            next = pred.get(ind).get(next);
        }

        return labels;
    }

    /**
     * Console-based test method that receives a line from the user and outputs the line with tags
     */
    public void tagLine() {
        //Scanner to read input
        Scanner scan = new Scanner(System.in);
        System.out.println("--\nEnter a sentence or type 'q' to quit: ");

        String line = scan.nextLine();

        while (!line.equals("q")) {
            List<String> tags = viterbi(line); //pass inputted line to viterbi method to get a list of tags
            List<String> words = Arrays.asList(line.split("\\s"));

            //print out tags with words in sentence
            for (int i = 0; i < words.size(); i++) {
                System.out.print(words.get(i) + "/" + tags.get(i) + " ");
            }

            System.out.println("\n\nEnter a sentence or type 'q' to quit: ");
            line = scan.nextLine();
        }
    }

    /**
     * File-based test method that evaluates performance of POSTagger given a pair of test files: one with sentences
     * and one with tags
     *
     * @param testSentPathName test sentences
     * @param testTagPathName test tags
     */
    public void testTagger(String testSentPathName, String testTagPathName) {

        // Sets up scanner
        BufferedReader sentInput = null, tagInput = null;

        try {
            sentInput = new BufferedReader(new FileReader(testSentPathName));
            tagInput = new BufferedReader(new FileReader(testTagPathName));
        }
        catch (FileNotFoundException e) {
            System.err.println("Cannot open file.\n" + e.getMessage());
        }

        try {

            // Gets sentence and corresponding tags from files
            String sentNext = sentInput.readLine();
            String tagNext = tagInput.readLine();

            int numMatching = 0;
            int numTotal = 0;

            // While there are lines to read, increments the matching tags and total tags sentence by sentence
            while (sentNext != null && tagNext != null) {
                String[] expectedTags = tagNext.split("\\s");
                List<String> predictedTags = viterbi(sentNext);

                for (int i = 0; i < expectedTags.length; i++) {
                    if (expectedTags[i].equals(predictedTags.get(i))) { numMatching++; }
                }

                numTotal+=expectedTags.length;

                sentNext = sentInput.readLine();
                tagNext = tagInput.readLine();
            }

            // Outputs performance using the number of correct and total tags
            System.out.println("\nEvaluating performance for '" + testSentPathName + "' and '" + testTagPathName +"':");
            System.out.println(numMatching + " correct, " + (numTotal-numMatching) + " wrong (" + Math.round((double)numMatching/numTotal*100.0)+"% accuracy)");
        }
        catch (IOException e) {
            System.err.println("IO error while reading.\n" + e.getMessage());
        }
    }

    /**
     * Setter method for probability maps that is used to test the viterbi algorithm with hard-coded
     * graphs
     */
    public void setProb(HashMap<String, HashMap<String, Double>> tagProb, HashMap<String, HashMap<String, Double>> obsProb){
        this.tagProb = tagProb;
        this.obsProb = obsProb;
    }

    /**
     * Setter method for training files
     * @param sentFile sentence training file
     * @param tagFile tags training file
     */
    public void setTrainingData(String sentFile, String tagFile) {
        trainSentPathName = sentFile;
        trainTagPathName = tagFile;
    }

    /**
     * Console-based test method that receives a line from the user and outputs the suggested best next word
     */
    public void bestNextWord() {

        // Sets up scanner
        Scanner scan = new Scanner(System.in);

        System.out.println("--\nEnter a sentence/phrase WITHOUT a period, and a next word will be suggested (type 'q' to quit):");

        String line = scan.nextLine();
        line = line.toLowerCase();

        // until 'q' is entered, keep reading next line and suggesting the next word for each inputted line
        while (!line.equals("q")) {
            ArrayList<String> sent = new ArrayList<String>(Arrays.asList(line.split("\\s")));
            ArrayList<String> labels = new ArrayList<String>();
            ArrayList<HashMap<String, String>> pred = new ArrayList<HashMap<String, String>>();
            sent.add(0, "#");

            ArrayList<String> currStates = new ArrayList<String>();
            currStates.add("#");

            HashMap<String, Double> currScores = new HashMap<String, Double>();
            currScores.put("#", 0.0);

            //handle sentence
            for (int ind = 0; ind < sent.size() - 1; ind++) {
                ArrayList<String> nextStates = new ArrayList<String>();
                HashMap<String, Double> nextScores = new HashMap<String, Double>();
                pred.add(new HashMap<String, String>());

                for (int curInd = 0; curInd < currStates.size(); curInd++) {
                    for (HashMap.Entry<String, Double> trans : tagProb.get(currStates.get(curInd)).entrySet()) {
                        if (!nextStates.contains(trans.getKey()))
                            nextStates.add(trans.getKey());

                        //calculate observed score
                        double obsScore = 0.0;
                        if (obsProb.get(trans.getKey()).containsKey(sent.get(ind + 1)))
                            obsScore = obsProb.get(trans.getKey()).get(sent.get(ind + 1));
                        else
                            obsScore = UNOBSERVED;

                        double nextScore = currScores.get(currStates.get(curInd)) + trans.getValue() + obsScore;

                        if (!nextScores.containsKey(trans.getKey()) || nextScore > nextScores.get(trans.getKey())) {
                            nextScores.put(trans.getKey(), nextScore);
                            pred.get(ind).put(trans.getKey(), currStates.get(curInd));
                        }
                    }
                }
                currStates = nextStates;
                currScores = nextScores;
            }

            //backtrack to identify path
            double maxScore = Double.MAX_VALUE * (-1.0);
            String maxKey = "";
            for (HashMap.Entry<String, Double> score : currScores.entrySet()) {
                if (score.getValue() > maxScore) {
                    maxScore = score.getValue();
                    maxKey = score.getKey();
                }
            }

            // Gets the transition with the highest probability from the last tag in the sentence
            String nextPOS = Collections.max(tagProb.get(maxKey).entrySet(), (HashMap.Entry<String, Double> e1, HashMap.Entry<String, Double> e2) -> e1.getValue()
                    .compareTo(e2.getValue())).getKey();

            // Gets the word with the highest probability for the predicted next part of speech
            String nextWord = Collections.max(obsProb.get(nextPOS).entrySet(), (HashMap.Entry<String, Double> e1, HashMap.Entry<String, Double> e2) -> e1.getValue()
                    .compareTo(e2.getValue())).getKey();

            // Outputs suggested next word
            System.out.println("Suggested next word: " + nextWord);

            System.out.println("\nEnter a sentence/phrase WITHOUT a period, and a next word will be suggested (type 'q' to quit):");
            line = scan.nextLine();
        }
    }

}