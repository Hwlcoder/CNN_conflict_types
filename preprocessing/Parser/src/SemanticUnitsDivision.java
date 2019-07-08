import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.io.*;
import javax.swing.JOptionPane;
import java.io.StringReader;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.SentenceUtils;
import edu.stanford.nlp.trees.*;
import edu.stanford.nlp.trees.international.pennchinese.ChineseTreebankLanguagePack;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import edu.stanford.nlp.process.CoreLabelTokenFactory;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.process.Tokenizer;
import edu.stanford.nlp.process.TokenizerFactory;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.tregex.TregexMatcher;
import edu.stanford.nlp.trees.tregex.TregexPattern;
import edu.stanford.nlp.trees.tregex.tsurgeon.Tsurgeon;
import edu.stanford.nlp.trees.tregex.tsurgeon.TsurgeonPattern;


public class SemanticUnitsDivision {
	//String parserModel = "edu/stanford/nlp/models/lexparser/chinesePCFG.ser.gz";
	String parserModel = "edu/stanford/nlp/models/lexparser/chineseFactored.ser.gz";
	//String parserModel = "edu/stanford/nlp/models/lexparser/xinhuaFactored.ser.gz";

	LexicalizedParser lp = LexicalizedParser.loadModel(parserModel);
	Tree parser;
	TreebankLanguagePack tlp = new ChineseTreebankLanguagePack();
	GrammaticalStructureFactory gsf = tlp.grammaticalStructureFactory();
	GrammaticalStructure gs ;
	List<TypedDependency> tdl;
	String full="";//原句
	String subject="";
	String action="";
	String consequence="";

    public void parserToTree(String sentence){//输入分词后的句子
       TokenizerFactory<CoreLabel> tokenizerFactory =
       PTBTokenizer.factory(new CoreLabelTokenFactory(), "");
       Tokenizer<CoreLabel> tok =
    	        tokenizerFactory.getTokenizer(new StringReader(sentence));
       List<CoreLabel> rawWords = tok.tokenize();
       parser = lp.apply(rawWords);
       parser.pennPrint();//输出树
       for (Tree  leaf : parser.getLeaves()) {
			  full +=leaf.value()+" ";
		 }
       System.out.println(full);
	}
    
    
	public String extractMainaction() {  //提取谓语中心词
		String mainAction="";
		gs= gsf.newGrammaticalStructure(parser);
	    tdl = gs.typedDependenciesCCprocessed();
	    System.out.println(tdl);
	    for(TypedDependency td:tdl) {
	       if(td.reln().getShortName()=="root") {
	    	   mainAction=td.dep().value();
	       }
	     }
	    System.out.println("谓语是:"+mainAction);
	   // extractAction(mainAction);
	    return mainAction;
	}
	public String extractSubject(String mainAction) { //提取主体
		 String mainSubject="";
		
		 for(TypedDependency td:tdl) {
		       if((td.reln().getShortName()=="nsubj")&&(td.gov().value().equals(mainAction))) {
		    	      mainSubject=td.dep().value();
		    	      System.out.println("主语是:"+mainSubject);
		    	 	 // TreeGraphNode dep = td.dep();
		           }
		 }
		 if (mainSubject=="") {
			 System.out.println("主体是：空");
		 }
		 else {
		     String s = "NP>IP&<<"+mainSubject; //np是ip的下一层节点，且np子树中包含主语
		     TregexPattern p = TregexPattern.compile(s);
	         TregexMatcher m = p.matcher(parser);   
	         while(m.find()) {
					 for (Tree  leaf : m.getMatch().getLeaves()) {
						  subject +=leaf.value()+" ";
					 }
			 }
		     System.out.println("主体是:"+subject);
	     }
		 return subject;
	}

	public String extractConsequence() { //提取后果
		 String[] trigger= {"奖励","警告","罚款","罚金","暂扣","吊销","拘留","拘役","刑事责任","处分","行政处分","取消警衔","赔偿",
				 "有期徒刑","无期徒刑","死刑","管制","驱逐出境","没收","处罚","行政处罚","降职","撤职","开除","责令","民事责任","撤销","行政责任","处罚金","调离",
				 "停止侵害","排除妨碍","消除危险","返还财产","恢复原状","修理","重作","更换","违约金","消除影响","恢复名誉","赔礼道歉"};
		 int i=0;
		 for(String x :trigger) {
			String s = "VP>(VP>IP)&<<"+x; 
			 TregexPattern p = TregexPattern.compile(s);
		     TregexMatcher m = p.matcher(parser);   
		     while(m.find()) {
						 for (Tree  leaf : m.getMatch().getLeaves()) {
							 consequence +=leaf.value()+" ";
						 }
						i++;
				 }
		     
		 }
		 //System.out.println("i="+i);
	     if(consequence=="") {
	    	 action=remove(full,subject);
	    	 System.out.println("行为是:"+action);
	    	 System.out.println("后果是：空");
	     }
	     else {
	    	 if(i>=2) {
	    		 System.out.println(consequence);
	    		 consequence=removeRepeat(consequence);
	    	 }
	    	 action=remove(full,subject);
	    	 action=remove(action,consequence);
	    	 System.out.println("行为是:"+action);
	    	 System.out.println("后果是:"+consequence);
	     }
		
		 return consequence;
	}
	public String getAction() {
		return action;
	}
	public String removeRepeat(String s) {//字符串去重
		Set<String> mLinkedSet = new LinkedHashSet<String>();
		String[] strArray = s.split(" ");
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < strArray.length; i++) {
			if (!mLinkedSet.contains(strArray[i])) {
				mLinkedSet.add(strArray[i]);
				sb.append(strArray[i] + " ");
			}
		}
		return sb.toString();
    }
	
	public String remove(String s1,String s2) {//字符串相减
		Set<String> mLinkedSet = new LinkedHashSet<String>();
		String[] strArray1 = s1.split(" ");
        String[] strArray2 = s2.split(" ");
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < strArray2.length; i++) {
		      mLinkedSet.add(strArray2[i]);
		}
        for (int j = 0; j < strArray1.length; j++) {
            if(!mLinkedSet.contains(strArray1[j])){
		      sb.append(strArray1[j] + " ");
             }
		}
		return sb.toString();
    }

	
	  public static void main(String[] args) {
		  String filePath="F:\\myCode\\Parser\\data\\cut_300\\Subject.txt";
	      File file=new File(filePath);
	      File f1=new File("F:\\myCode\\Parser\\data\\result_300\\Subject_parse.txt");
	      if(file.isFile() && file.exists())
	        { 
	    	  try {
	            InputStreamReader read = new InputStreamReader(new FileInputStream(file));
	            BufferedReader bufferedReader = new BufferedReader(read);
	            FileOutputStream fos = new FileOutputStream(f1);
	            String lineTxt = null;
	            long startTime = System.currentTimeMillis(); //获取开始时间
	            while((lineTxt = bufferedReader.readLine())!= null)
	              {
	            	 System.out.println(lineTxt);
	            	 String mainAction="";
	       		     String Subject="";
	       		     String Action="";
	       		     String Consequence="";
	       		     SemanticUnitsDivision extract=new SemanticUnitsDivision();
	       		     extract.parserToTree(lineTxt);
		       		 mainAction=extract.extractMainaction();
		       		 Subject=extract.extractSubject(mainAction);
		       		 Consequence=extract.extractConsequence();
		       		 Action=extract.getAction();
		       		 StringBuffer sb=new StringBuffer();
		       		 sb.append(Subject+"\r\n"+Action+"\r\n"+Consequence+"\r\n");
		       		 fos.write(sb.toString().getBytes("utf-8"));
	              }
	            long endTime = System.currentTimeMillis(); //获取结束时间
	            System.out.println("程序运行时间：" + (endTime - startTime)/1000+ "s"); //输出程序运行时间
	            bufferedReader.close();
	            read.close();
	            fos.close();
	    	  } catch (Exception ex) {
     	            JOptionPane.showMessageDialog(null, ex.getStackTrace());
     	        }
	        } 
	    	  } 
}
