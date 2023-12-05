package com.example.scorereader;

import android.content.Intent;
import android.media.AudioManager;
import android.media.MediaPlayer;
import android.media.SoundPool;
import android.os.Build;
import android.os.Bundle;
import android.view.MotionEvent;
import android.view.View;
import android.view.accessibility.AccessibilityEvent;
import android.view.accessibility.AccessibilityNodeInfo;
import android.widget.Button;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

public class Score1 extends AppCompatActivity  {

    int i = 0;
    int position = 0;
    int sFlag = 0;

    String scorelist[]= new String[] {
            "4분의1음표4옥타브 도입니다.", "4분의1음표4옥타브 도입니다", "4분의1음표4옥타브 솔입니다", "4분의1음표4옥타브 솔입니다","4분의1음표4옥타브 라입니다", "4분의1음표4옥타브 라입니다", "2분의1음표4옥타브 솔입니다", "4분의1음표4옥타브 파입니다","4분의1음표4옥타브 파입니다", "4분의1음표4옥타브 미입니다", "4분의1음표4옥타브 미입니다", "4분의1음표4옥타브 레입니다","4분의1음표4옥타브 레입니다", "2분의1음표4옥타브 도입니다", "4분의1음표4옥타브 솔입니다", "4분의1음표4옥타브 솔입니다","4분의1음표4옥타브 파입니다", "4분의1음표4옥타브 파입니다", "4분의1음표4옥타브 미입니다", "4분의1음표4옥타브 미입니다","2분의1음표4옥타브 레입니다", "4분의1음표4옥타브 솔입니다",
            "4분의1음표4옥타브 솔입니다", "4분의1음표4옥타브 파입니다","4분의1음표4옥타브 파입니다", "4분의1음표4옥타브 미입니다", "4분의1음표4옥타브 미입니다", "8분의3음표4옥타브 레입니다","8분의3음표4옥타브 도입니다", "8분의1음표4옥타브 도입니다", "8분의3음표4옥타브 솔입니다", "8분의1음표4옥타브 솔입니다","8분의3음표4옥타브 라입니다", "8분의1음표4옥타브 라입니다", "4분의1음표4옥타브 라입니다", "4분의1음표4옥타브 솔입니다","8분의3음표4옥타브 파입니다", "8분의1음표4옥타브 파입니다", "8분의3음표4옥타브 미입니다", "8분의1음표4옥타브 미입니다","8분의1음표4옥타브 레입니다", "8분의1음표4옥타브 도입니다", "8분의1음표4옥타브 레입니다", "8분의1음표4옥타브 미입니다",
            "8분의1음표4옥타브 도입니다", "8분의1음표4옥타브 도입니다", "8분의1음표4옥타브 레입니다", "8분의1음표4옥타브 도입니다","8분의1음표4옥타브 레입니다", "8분의1음표4옥타브 솔입니다", "8분의1음표4옥타브 솔입니다", "8분의1음표4옥타브 솔입니다","8분의1음표4옥타브 솔입니다", "8분의1음표4옥타브 라입니다", "8분의1음표4옥타브 라입니다", "8분의1음표4옥타브 라입니다","8분의1음표4옥타브 라입니다", "2분의1음표4옥타브 솔입니다", "8분의1음표4옥타브 파입니다", "8분의1음표4옥타브 파입니다","8분의1음표4옥타브 파입니다", "8분의1음표4옥타브 파입니다", "4분의1음표4옥타브 미입니다", "4분의1음표4옥타브 파입니다","8분의1음표4옥타브 레입니다", "8분의1음표4옥타브 레입니다",
            "8분의1음표4옥타브 레입니다", "8분의1음표4옥타브 레입니다","2분의1음표4옥타브 도입니다", "8분의1음표4옥타브 솔입니다", "8분의1음표4옥타브 솔입니다", "8분의1음표4옥타브 솔입니다","8분의1음표4옥타브 솔입니다", "4분의1음표4옥타브 파입니다", "8분의1음표4옥타브 파입니다",  "8분의1음표4옥타브 미입니다","8분의1음표4옥타브 미입니다", "8분의1음표4옥타브 미입니다", "8분의1음표4옥타브 미입니다",  "8분의1음표4옥타브 미입니다","4분의1음표4옥타브 레입니다", "4분의1음표4옥타브 레입니다", "8분의1음표4옥타브 솔입니다",  "8분의1음표4옥타브 솔입니다","8분의1음표4옥타브 솔입니다", "8분의1음표4옥타브 솔입니다", "16분의1음표4옥타브 파입니다", "16분의1음표4옥타브 파입니다",
            "16분의1음표4옥타브 파입니다", "16분의1음표4옥타브 파입니다", "16분의1음표4옥타브 파입니다", "16분의1음표4옥타브 파입니다","16분의1음표4옥타브 파입니다", "16분의1음표4옥타브 파입니다", "8분의1음표4옥타브 미입니다", "8분의1음표4옥타브 미입니다","8분의1음표4옥타브 미입니다", "1분의1음표4옥타브 레입니다", "4분의1음표4옥타브 레입니다", "4분의1음표4옥타브 레입니다."

    };

    double posList[]=new double[]{
            0, 0.5, 1.0, 1.5,
            2.0, 2.5, 3.0, 4.0,
            4.5, 5.0, 5.5, 6.0,
            6.5, 7.0, 8.0, 8.5,
            9.0, 9.5, 10.0, 10.5,
            11.0, 12.0, 12.5, 13.0,
            13.5, 14.0, 14.5, 15.0,
            15.75, 16.5, 16.75, 17.5,
            17.75, 18.5, 18.75, 19.25,
            19.75, 20.5, 20.75, 21.5,
            21.75, 22.0, 22.25, 22.5,
            22.75, 23.0, 23.25, 23.5,
            23.75, 24.0, 24.25, 24.5,
            24.75, 25.0, 25.25, 25.5,
            25.75, 26.0, 27.0, 27.25,
            27.5, 27.75, 28.0, 28.5,
            29.0, 29.25, 29.5, 29.75,
            30.0, 31.0, 31.25, 31.5,
            31.75, 32.0, 32.5, 32.75,
            33.0, 33.25, 33.5, 33.75,
            34.0, 34.5, 35.0, 35.25,
            35.5, 35.75, 36.0, 36.125,
            36.25, 36.375, 36.5, 36.625,
            36.75, 36.875, 37.0, 37.25,
            37.5, 37.75, 38.0, 38.5,
            39.0 };



    MediaPlayer mediaPlayer;

    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.printscore);
//        View left = (View)findViewById(R.id.view1);
//        View right = (View)findViewById(R.id.view2);
        Button before= (Button)findViewById(R.id.button1);
        Button after = (Button)findViewById(R.id.button2);
        Button start= (Button)findViewById(R.id.button3);
        Button stop = (Button)findViewById(R.id.button4);
        TextView score = (TextView)findViewById(R.id.textView);

        mediaPlayer = MediaPlayer.create(this, R.raw.star);

        PlaySound playSound = new PlaySound();





        int freqList[]=new int[scorelist.length];

        for (int j = 0; j<scorelist.length; j++){
            if(scorelist[j].contains("도")){
                freqList[j]=262;
            }
            else if(scorelist[j].contains("레")){
                freqList[j]=294;
            }
            else if(scorelist[j].contains("미")){
                freqList[j]=330;
            }
            else if(scorelist[j].contains("파")){
                freqList[j]=349;
            }
            else if(scorelist[j].contains("솔")){
                freqList[j]=392;
            }
            else if(scorelist[j].contains("라")){
                freqList[j]=440;
            }
            else if(scorelist[j].contains("시")){
                freqList[j]=494;
            }
            else{
                freqList[j]=440;
            }

        }

        score.setText(scorelist[i]);

        before.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                mediaPlayer.pause();
                sFlag = 0;
                start.setText("재생");

                if(i > 0){
                    i--;
                    score.setText(scorelist[i]);
                    playSound.setFreqOfTone(freqList[i]);
                }
                if(i > scorelist.length-1){
                    i = 0;
                    score.setText(scorelist[i]);
                    playSound.setFreqOfTone(freqList[i]);
                }else if(i==0){
                    i = scorelist.length-1;
                    score.setText(scorelist[i]);
                    playSound.setFreqOfTone(freqList[i]);

                }

            }
        });

        after.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                mediaPlayer.pause();
                sFlag = 0;
                start.setText("재생");

                if(scorelist.length-1 > i & i >= 0){
                    i++;
                    score.setText(scorelist[i]);
                    playSound.setFreqOfTone(freqList[i]);
                }
                if(i >= scorelist.length-1){
                    i = 0;
                    score.setText(scorelist[i]);
                    playSound.setFreqOfTone(freqList[i]);

                }else if(i<0){
                    i = scorelist.length-1;
                    score.setText(scorelist[i]);
                    playSound.setFreqOfTone(freqList[i]);
                }

            }
        });


        start.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if(sFlag ==0){
                    mediaPlayer.seekTo((int) posList[i]*1000);
                    mediaPlayer.start();
                    score.setText(" ");
                    sFlag = 1;
                    start.setText("멈춤");
                }
                else{
                    position=mediaPlayer.getCurrentPosition();
                    sortTime(position);
                    mediaPlayer.pause();
                    score.setText(scorelist[i]);
                    sFlag = 0;
                    start.setText("재생");
                }

            }
        });

        stop.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                mediaPlayer.pause();
                sFlag = 0;

                start.setText("멈춤");
                position=0;
                sortTime(position);
                mediaPlayer.seekTo((int) posList[i]*1000);
                mediaPlayer.start();
                score.setText(scorelist[i]);
                score.setText(" ");

            }
        });




//        left.setOnTouchListener(new View.OnTouchListener() {
//            public boolean onTouch(View v, MotionEvent event) {
//
//                switch (event.getAction()) {
//                    case MotionEvent.ACTION_DOWN: {
//                        left.sendAccessibilityEvent(AccessibilityEvent.TYPE_VIEW_FOCUSED);
//                        return false;
//                    }
//                    case MotionEvent.ACTION_UP: {
//                        if(i > 0){
//                            i--;
//                            count.setText(scorelist[i]);
//                            playSound.setFreqOfTone(freqList[i]);
//                        }
//                        else if(i > scorelist.length-1){
//                            i = 0;
//                            count.setText(scorelist[i]);
//                            playSound.setFreqOfTone(freqList[i]);
//                        }else if(i==0){
//                            i = scorelist.length-1;
//                            count.setText(scorelist[i]);
//                            playSound.setFreqOfTone(freqList[i]);
//
//                        }
//
//                    }
//
//                    default:
//                        return false;
//                }
//
//
//            }
//
//        });
//
//
//        right.setOnTouchListener(new View.OnTouchListener() {
//            public boolean onTouch(View v, MotionEvent event) {
//
//                switch (event.getAction()) {
//                    case MotionEvent.ACTION_DOWN: {
//                        return false;
//                    }
//                    case MotionEvent.ACTION_UP: {
//                        if(scorelist.length-1 > i & i >= 0){
//                            i++;
//                            count.setText(scorelist[i]);
//                            playSound.setFreqOfTone(freqList[i]);
//                        }
//                        else if(i > scorelist.length-1){
//                            i = 0;
//                            count.setText(scorelist[i]);
//                            playSound.setFreqOfTone(freqList[i]);
//
//                        }else if(i<0){
//                            i = scorelist.length-1;
//                            count.setText(scorelist[i]);
//                            playSound.setFreqOfTone(freqList[i]);
//                        }
//
//                    }
//                    default:
//                        return false;
//                }
//
//            }
//
//        });
//
    }
    public void sortTime(int position){
        int k;
        for(k=0; k<posList.length; k++){
            if( (int) posList[k] == (int) position/1000 ){
                i = k;
            }
        }
    }


}