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

def setup_cpu_library_functions(library):
    """CPU 라이브러리 함수 시그니처 설정"""
    if library is not None:
        # CPU 전용 단일파일 데이터 생성
        library.genGOLdataInOneFile.argtypes = [ct.c_uint32, ct.c_uint32, ct.c_double, ct.c_char_p]
        library.genGOLdataInOneFile.restype = None
        
        # CPU 패턴 예측 함수
        library.getPredict.argtypes = [ct.POINTER(ct.c_int)]
        library.getPredict.restype = ct.c_int

def setup_gpu_library_functions(library):
    """GPU 라이브러리 함수 시그니처 설정"""
    if library is not None:
        # gpu-cpu 오버해드 해결한 함수-단일 파일 버전.
        library.genGOLdataOptimizeInOneFile.argtypes = [ct.c_uint32, ct.c_uint32, ct.c_double]
        library.genGOLdataOptimizeInOneFile.restype = None

        #오버헤드 단순화 + 생성루트 지정
        library.genGOLdataOptimizeInOneFile_RootFlexible.argtypes = [ct.c_uint32, ct.c_uint32, ct.c_double, ct.c_char_p]
        library.genGOLdataOptimizeInOneFile_RootFlexible.restype = None
        
        # GPU 패턴 예측 함수
        library.getPredict.argtypes = [ct.POINTER(ct.c_int)]
        library.getPredict.restype = ct.c_int

def main():
    global lib
    
    parser = argparse.ArgumentParser(description='Game of Life 데이터 생성기')
    parser.add_argument('param1', type=int, help='첫 번째 매개변수 (uint32)')
    parser.add_argument('param2', type=int, help='두 번째 매개변수 (uint32)')
    parser.add_argument('param3', type=float, help='세 번째 매개변수 (double)')
    
    # 선택적 인자들
    parser.add_argument('--verbose', '-v', action='store_true', help='상세 출력')
    parser.add_argument('--output', '-o', type=str, help='출력 디렉토리')
    parser.add_argument('--cpu', action='store_true', help="외장 글카 없는 경우에 cpu로 데이터 생성이 가능케 함")
    
    args = parser.parse_args()
    
    if args.verbose:
        print(f"매개변수: {args.param1}, {args.param2}, {args.param3}")
        if args.output:
            print(f"출력 디렉토리: {args.output}")

    # 라이브러리 로드
    is_cpu_mode = args.cpu
    mode_str = "CPU" if is_cpu_mode else "GPU"
    
    lib_path = find_library(cpu_mode=is_cpu_mode)
    if lib_path is None:
        if not is_cpu_mode:
            print("💡 --cpu 옵션을 사용하여 CPU 모드로 시도해보세요.")
        return 1

    try:
        lib = ct.CDLL(lib_path)
        if is_cpu_mode:
            setup_cpu_library_functions(lib)
        else:
            setup_gpu_library_functions(lib)
        print(f"✅ 라이브러리 로드 성공: {lib_path}")
    except Exception as e:
        print(f"❌ {mode_str} 라이브러리 로드 실패: {e}")
        if not is_cpu_mode:
            print("💡 --cpu 옵션을 사용하여 CPU 모드로 시도해보세요.")
        return 1

    # C 라이브러리 함수 호출
    try:
        func_to_call = None
        if is_cpu_mode:
            func_to_call = lib.genGOLdataInOneFile
        else:  # GPU 모드
            if args.output:
                func_to_call = lib.genGOLdataOptimizeInOneFile_RootFlexible
            else:
                func_to_call = lib.genGOLdataOptimizeInOneFile
        
        if func_to_call is None:
            print("❌ 호출할 함수를 찾을 수 없습니다.")
            return 1

        # 함수 호출
        output_path = args.output.encode('utf-8')
        func_to_call(args.param1, args.param2, args.param3, ct.c_char_p(output_path))

    except Exception as e:
        print(f"❌ {mode_str} 모드 오류: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
