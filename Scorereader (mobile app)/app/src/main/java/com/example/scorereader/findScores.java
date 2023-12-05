package com.example.scorereader;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import androidx.appcompat.app.AppCompatActivity;

public class findScores extends AppCompatActivity {


    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.scorelist);
        Button step1 = (Button) findViewById(R.id.button1);
        Button step2 = (Button) findViewById(R.id.button2);
        Button step3 = (Button) findViewById(R.id.button3);
        Button step4 = (Button) findViewById(R.id.button4);

        //아직 악보 1만 접근 가능
        step1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                //score1 페이지 이동
                Intent intent = new Intent(getApplicationContext(), Score1.class);
                startActivity(intent);
            }
        });

        step2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                //score1 페이지 이동
                Intent intent = new Intent(getApplicationContext(), Score2.class);
                startActivity(intent);
            }
        });
        step3.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                //score1 페이지 이동
                Intent intent = new Intent(getApplicationContext(), Score3.class);
                startActivity(intent);
            }
        });

    }

}
