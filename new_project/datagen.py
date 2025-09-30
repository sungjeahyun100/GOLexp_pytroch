# -*- coding: utf-8 -*-
import ctypes as ct
import os
import argparse
import sys

def find_library(cpu_mode=False):
    """공유 라이브러리를 여러 경로에서 찾기"""
    if cpu_mode:
        # CPU 모드용 라이브러리 경로
        possible_paths = [
            '../build/GOLdatagen_cpu.so',
            './build/GOLdatagen_cpu.so', 
            'build/GOLdatagen_cpu.so',
            os.path.join(os.path.dirname(__file__), '..', 'build', 'GOLdatagen_cpu.so')
        ]
    else:
        # GPU 모드용 라이브러리 경로
        possible_paths = [
            '../build/GOLdatagen_gpu.so',
            './build/GOLdatagen_gpu.so', 
            'build/GOLdatagen_gpu.so',
            os.path.join(os.path.dirname(__file__), '..', 'build', 'GOLdatagen_gpu.so')
        ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    mode_str = "CPU" if cpu_mode else "GPU"
    print(f"❌ {mode_str} 공유 라이브러리를 찾을 수 없습니다!")
    print("다음 경로들을 확인했습니다:")
    for path in possible_paths:
        print("  - " + os.path.abspath(path))
    print("\n해결 방법:")
    print("1. 프로젝트 루트에서 'mkdir build && cd build && cmake .. && make' 실행")
    print("2. CUDA가 없는 경우 --cpu 옵션 사용")
    return None

# 전역 변수로 lib 초기화 (나중에 main에서 설정)
lib = None
cpu_lib = None
gpu_lib = None

def setup_cpu_library_functions(library):
    """CPU 라이브러리 함수 시그니처 설정"""
    if library is not None:
        # CPU 전용 호스트 데이터 생성
        library.genGOLdataInHost.argtypes = [ct.c_uint32, ct.c_uint32, ct.c_double]
        library.genGOLdataInHost.restype = None

        # CPU 전용 단일파일 데이터 생성
        library.genGOLdataInOneFile.argtypes = [ct.c_uint32, ct.c_uint32, ct.c_double]
        library.genGOLdataInOneFile.restype = None
        
        # CPU 패턴 예측 함수
        library.getPredict.argtypes = [ct.POINTER(ct.c_int)]
        library.getPredict.restype = ct.c_int

def setup_gpu_library_functions(library):
    """GPU 라이브러리 함수 시그니처 설정"""
    if library is not None:
        # GPU 멀티파일 데이터 생성
        library.genGOLdata.argtypes = [ct.c_uint32, ct.c_uint32, ct.c_double]
        library.genGOLdata.restype = None

        # GPU 단일파일 데이터 생성
        library.genGOLdataInOneFile.argtypes = [ct.c_uint32, ct.c_uint32, ct.c_double]
        library.genGOLdataInOneFile.restype = None
        
        # GPU 패턴 예측 함수
        library.getPredict.argtypes = [ct.POINTER(ct.c_int)]
        library.getPredict.restype = ct.c_int

def main():
    global lib, cpu_lib, gpu_lib
    
    parser = argparse.ArgumentParser(description='Game of Life 데이터 생성기')
    parser.add_argument('param1', type=int, help='첫 번째 매개변수 (uint32)')
    parser.add_argument('param2', type=int, help='두 번째 매개변수 (uint32)')
    parser.add_argument('param3', type=float, help='세 번째 매개변수 (double)')
    
    # 선택적 인자들
    parser.add_argument('--verbose', '-v', action='store_true', help='상세 출력')
    parser.add_argument('--output', '-o', type=str, help='출력 디렉토리')
    parser.add_argument('--one_file', action='store_true', help='한 파일에 데이터를 몰아서 저장')
    parser.add_argument('--cpu', action='store_true', help="외장 글카 없는 경우에 cpu로 데이터 생성이 가능케 함")
    
    args = parser.parse_args()
    
    if args.verbose:
        print("매개변수: {}, {}, {}".format(args.param1, args.param2, args.param3))
        if args.output:
            print("출력 디렉토리: " + args.output)
    
    # CPU 모드인지 GPU 모드인지에 따라 다른 라이브러리 로드
    if args.cpu:
        # CPU 모드: cpu 라이브러리 로드
        cpu_lib_path = find_library(cpu_mode=True)
        if cpu_lib_path is None:
            print("❌ CPU 라이브러리를 사용할 수 없습니다!")
            return 1
        try:
            cpu_lib = ct.CDLL(cpu_lib_path)
            setup_cpu_library_functions(cpu_lib)
            lib = cpu_lib  # 현재 사용할 라이브러리 설정
            print("✅ 라이브러리 로드 성공: " + cpu_lib_path)
        except Exception as e:
            print("❌ CPU 라이브러리 로드 실패: " + str(e))
            return 1
    else:
        # GPU 모드: gpu 라이브러리 로드
        gpu_lib_path = find_library(cpu_mode=False)
        if gpu_lib_path is None:
            print("❌ GPU 라이브러리를 사용할 수 없습니다!")
            print("💡 --cpu 옵션을 사용하여 CPU 모드로 시도해보세요.")
            return 1
        try:
            gpu_lib = ct.CDLL(gpu_lib_path)
            setup_gpu_library_functions(gpu_lib)
            lib = gpu_lib  # 현재 사용할 라이브러리 설정
            print("✅ 라이브러리 로드 성공: " + gpu_lib_path)
        except Exception as e:
            print("❌ GPU 라이브러리 로드 실패: " + str(e))
            print("💡 --cpu 옵션을 사용하여 CPU 모드로 시도해보세요.")
            return 1
    
    # C 라이브러리 함수 호출
    if args.cpu:
        # CPU 모드: genGOLdataInHost 또는 genGOLdataInOneFile 사용
        try:
            if args.one_file:
                lib.genGOLdataInOneFile(args.param1, args.param2, args.param3)
            else:
                lib.genGOLdataInHost(args.param1, args.param2, args.param3)
        except Exception as e:
            print("❌ CPU 모드 오류: " + str(e))
            return 1
    else:
        # GPU 모드: genGOLdata 또는 genGOLdataInOneFile 사용
        try:
            if args.one_file:
               lib.genGOLdataInOneFile(args.param1, args.param2, args.param3)
            else:
               lib.genGOLdata(args.param1, args.param2, args.param3)
        except Exception as e:
            print("❌ GPU 모드 오류: " + str(e))
            return 1
        
        return 0

if __name__ == "__main__":
    sys.exit(main())
