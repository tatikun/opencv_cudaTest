#ifndef MYTIMER_H
#define MYTIMER_H

#include <Windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

/**
 * @brief   ���Ԍv������N���X
 */
class MyTimer
{
private:
	LARGE_INTEGER Freq;			// ���g��
	LARGE_INTEGER lastCount;	// �O��̃J�E���g
	LARGE_INTEGER thisCount;	// �ŐV�̃J�E���g

	int frameCount;				// �t���[���̃J�E���g
	double updateTime;			// fps�X�V�̊Ԋu(�b)
	double fixedFps;			// �Œ肵����fps�̒l
	double fps;					// fps
	bool fps_flag;				// FPS�̍X�V�̂��߂̃t���O


public:
	MyTimer() 
		: frameCount (0)
		, updateTime (1.0)
		, fixedFps (120.0)
		, fps (0.0)
		, fps_flag(false)
	{
		// ���g���̎擾
		QueryPerformanceFrequency( &Freq );
		start();	// ��������
	}
	~MyTimer() {}

	// �v���J�n
	inline void start()
	{
		QueryPerformanceCounter( &lastCount );		// �J�E���g�̕ۑ�
	}

	// �v���I��
	inline void stop()
	{
		QueryPerformanceCounter( &thisCount );		// �J�E���g�̕ۑ�
	}

	// �b�̌v��
	inline double Sec()
	{
		return (double)( thisCount.QuadPart - lastCount.QuadPart ) / Freq.QuadPart;
	}

	// �~���b�̌v��
	inline double MSec()
	{
		return 1000.0 * (double)( thisCount.QuadPart - lastCount.QuadPart ) / Freq.QuadPart;
	}

	// �b�̕\��
	inline void printSec()
	{
		std::cout << Sec() << "[sec]" << std::endl;
	}

	// �~���b�̕\��
	inline void printMSec()
	{
		std::cout << MSec() << "[msec]" << std::endl;
	}

	// FPS�̎擾
	inline double getFps() {	return fps; }

	// FPS�X�V�Ԋu�̃Z�b�g
	inline void setUpdateFps(double t)
	{
		updateTime = t;
	}

	// �Œ肵����FPS�̒l�̃Z�b�g
	inline void setFixedFps(double f) 
	{
		fixedFps = f;
	}

	// �w�肵��FPS�̌v��
	inline bool control_fps()
	{
		stop();

		// 1�t���[���ڂȂ�Ύ������L�^
		if ( frameCount == 0 ) 
		{
			fps_flag = false;
			start();
			frameCount++;
		}
		// fps�̍X�V���Ԃ�������t���O��true
		else if ( MSec() >= (1000 / fixedFps) )
		{
			fps_flag = true;
			frameCount = 0;
			start();
		}
		else{
			frameCount++;
		}		

		return fps_flag;
	}

	// FPS�v���̍X�V
	inline void fpsUpdate()
	{
		stop();

		// 1�t���[���ڂȂ�Ύ������L�^
		if ( frameCount == 0 ) 
		{
			start();	
		}
		// fps�̍X�V���Ԃ�������X�V
		else if ( Sec() >= updateTime )
		{
			fps = 1.0 / (Sec() / (double)frameCount);
			frameCount = 0;
			start();
		}

		frameCount++;
	}

	// FPS�̌Œ�
	inline void wait()
	{
		LARGE_INTEGER nowCount;
		QueryPerformanceCounter( &nowCount );
		double tookTime = 1000.0 * (double)( nowCount.QuadPart - lastCount.QuadPart ) / Freq.QuadPart;	// ������������
		double waitTime = (double)frameCount * 1000.0 / fixedFps - tookTime;					// �҂ׂ�����
		if(waitTime > 0.0)
			Sleep((DWORD)waitTime);	// �ҋ@
	}

	// FPS�̕\��
	inline void printFps(bool wait = true)
	{
		if( wait && frameCount != 1)
		{
			return;
		}
		std::cout << fps << "[fps]" << std::endl;
	}
};


#endif