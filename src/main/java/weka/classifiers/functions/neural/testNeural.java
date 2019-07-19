package weka.classifiers.functions.neural;


import java.util.Random;

public class testNeural {


    public static void main(String[] args) {
        NeuralMethod linear = new LinearUnit();
        NeuralNode x1 = new NeuralNode("x1",new Random(),linear);
        NeuralNode x2 = new NeuralNode("x2",new Random(),linear);
        NeuralNode x3 = new NeuralNode("x3",new Random(),linear);
//        NeuralNode x4 = new NeuralNode("x4",new Random(),linear);



        x3.connectInput(x1,1);  //x1作为x3的输入，随机初始化一个权重，将权重标记为未改变
        x3.connectInput(x2,2);

        x1.connectOutput(x3,3);
        x2.connectOutput(x3,3);


        System.out.println(linear.errorValue(x3));



        for(double a:x3.getWeights()){
            System.out.println("aaa:"+a);
        }

        x3.updateWeights(1000,0.1);


        System.out.println(x3.outputValue(true));


//        System.out.println(x3.weightValue(1));
//        System.out.println(x1.outputValue(true));
//        System.out.println(x3.outputValue(true));
//        System.out.println(x4.outputValue(true));
//
//
//        System.out.println(x3.weightValue(1));
//        System.out.println(x3.weightValue(0));


//
//        int []a = new int[0];
//        for(int b:a){
//            System.out.println(b);
//        }


//        x1.setX(1);x1.setY(1);
//        x2.setX(2);x2.setY(2);
//        x3.setX(3);x3.setY(3);

//        System.out.println(x1.getNumInputs());
//        System.out.println(x1.weightValue(0));








    }

}
