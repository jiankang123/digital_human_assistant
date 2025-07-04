#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数字人交互助手HTTPS服务器 - 通义千问版本
支持SSL证书自动生成和通义千问API代理
"""

import os
import sys
import ssl
import json
import argparse
import subprocess
from datetime import datetime, timedelta
from urllib.parse import urlparse, parse_qs
from http.server import HTTPServer, SimpleHTTPRequestHandler

# 尝试导入requests库
try:
    import requests
except ImportError:
    print("错误: requests库未安装")
    print("请运行: pip install requests")
    sys.exit(1)


class DigitalHumanHandler(SimpleHTTPRequestHandler):
    """自定义HTTP请求处理器"""
    
    def __init__(self, *args, **kwargs):
        self.qwen_config = {
            'api_key': os.getenv('DASHSCOPE_API_KEY', 'sk-6c9377d37f5e4197a00b713fe439c129'),
            'base_url': 'https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation',
            'timeout': 30
        }
        super().__init__(*args, **kwargs)
    
    def do_POST(self):
        """处理POST请求"""
        if self.path == '/api/qwen':
            self.handle_qwen_api()
        else:
            self.send_error(404, "Not Found")
    
    def handle_qwen_api(self):
        """处理通义千问API请求"""
        try:
            # 检查API密钥
            if not self.qwen_config['api_key']:
                self.send_json_response({
                    'error': {
                        'message': 'DASHSCOPE_API_KEY环境变量未设置，请设置您的通义千问API密钥',
                        'type': 'auth_error'
                    }
                }, 401)
                return
            
            # 读取请求体
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self.send_json_response({
                    'error': {'message': '请求体为空', 'type': 'invalid_request'}
                }, 400)
                return
            
            request_body = self.rfile.read(content_length).decode('utf-8')
            request_data = json.loads(request_body)
            
            # 验证请求数据
            if 'messages' not in request_data:
                self.send_json_response({
                    'error': {'message': '缺少messages字段', 'type': 'invalid_request'}
                }, 400)
                return
            
            # 构建通义千问API请求
            qwen_request = {
                'model': request_data.get('model', 'qwen-turbo'),
                'input': {
                    'messages': request_data['messages']
                },
                'parameters': {
                    'max_tokens': request_data.get('max_tokens', 1500),
                    'temperature': request_data.get('temperature', 0.7),
                    'top_p': request_data.get('top_p', 0.8),
                    'repetition_penalty': 1.1
                }
            }
            
            # 调用通义千问API
            headers = {
                'Authorization': f'Bearer {self.qwen_config["api_key"]}',
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 调用通义千问API...")
            
            response = requests.post(
                self.qwen_config['base_url'],
                json=qwen_request,
                headers=headers,
                timeout=self.qwen_config['timeout']
            )
            
            if response.status_code == 200:
                qwen_response = response.json()
                
                # 转换为OpenAI格式的响应
                if 'output' in qwen_response and 'text' in qwen_response['output']:
                    openai_response = {
                        'id': f"chatcmpl-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        'object': 'chat.completion',
                        'created': int(datetime.now().timestamp()),
                        'model': qwen_request['model'],
                        'choices': [{
                            'index': 0,
                            'message': {
                                'role': 'assistant',
                                'content': qwen_response['output']['text']
                            },
                            'finish_reason': 'stop'
                        }],
                        'usage': qwen_response.get('usage', {
                            'prompt_tokens': 0,
                            'completion_tokens': 0,
                            'total_tokens': 0
                        })
                    }
                    self.send_json_response(openai_response)
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] API调用成功")
                else:
                    self.send_json_response({
                        'error': {'message': 'API返回格式异常', 'type': 'api_error'}
                    }, 500)
            else:
                # API调用失败
                try:
                    error_data = response.json()
                    error_message = error_data.get('message', f'API调用失败 (状态码: {response.status_code})')
                except:
                    error_message = f'API调用失败 (状态码: {response.status_code})'
                
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] API调用失败: {error_message}")
                
                self.send_json_response({
                    'error': {
                        'message': error_message,
                        'type': 'api_error',
                        'code': response.status_code
                    }
                }, response.status_code)
        
        except json.JSONDecodeError:
            self.send_json_response({
                'error': {'message': 'JSON格式错误', 'type': 'invalid_request'}
            }, 400)
        except requests.RequestException as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 网络请求错误: {str(e)}")
            self.send_json_response({
                'error': {'message': f'网络请求错误: {str(e)}', 'type': 'network_error'}
            }, 500)
        except Exception as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 服务器内部错误: {str(e)}")
            self.send_json_response({
                'error': {'message': '服务器内部错误', 'type': 'internal_error'}
            }, 500)
    
    def send_json_response(self, data, status_code=200):
        """发送JSON响应"""
        response_body = json.dumps(data, ensure_ascii=False, indent=2).encode('utf-8')
        
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(response_body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.end_headers()
        self.wfile.write(response_body)
    
    def do_OPTIONS(self):
        """处理预检请求"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.send_header('Content-Length', '0')
        self.end_headers()
    
    def do_GET(self):
        """处理GET请求"""
        if self.path == '/api/status':
            self.send_json_response({
                'status': 'running',
                'timestamp': datetime.now().isoformat(),
                'version': '2.0.0',
                'features': ['https', 'qwen-api', 'cors'],
                'qwen_api_configured': bool(self.qwen_config['api_key'])
            })
        elif self.path == '/api/qwen':
            # 显示API使用说明
            self.send_json_response({
                'message': '通义千问API端点',
                'description': '这是一个POST端点，用于与通义千问AI模型对话',
                'usage': {
                    'method': 'POST',
                    'url': '/api/qwen',
                    'headers': {
                        'Content-Type': 'application/json'
                    },
                    'body': {
                        'model': 'qwen-turbo',
                        'messages': [
                            {
                                'role': 'user',
                                'content': '你好'
                            }
                        ],
                        'max_tokens': 1500,
                        'temperature': 0.7
                    }
                },
                'example': 'curl -X POST http://localhost:8080/api/qwen -H "Content-Type: application/json" -d \'{"messages":[{"role":"user","content":"你好"}]}\'',
                'api_configured': bool(self.qwen_config['api_key']),
                'supported_models': ['qwen-turbo', 'qwen-plus', 'qwen-max']
            })
        else:
            # 处理静态文件请求
            super().do_GET()
    
    def log_message(self, format, *args):
        """自定义日志格式"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] {format % args}")


def create_ssl_certificate(cert_file='server.crt', key_file='server.key'):
    """创建自签名SSL证书"""
    if os.path.exists(cert_file) and os.path.exists(key_file):
        print(f"SSL证书已存在: {cert_file}, {key_file}")
        return True
    
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        import ipaddress
        
        print("正在生成SSL证书...")
        
        # 生成私钥
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        
        # 创建证书主题
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "CN"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Beijing"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "Beijing"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Digital Human Assistant"),
            x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
        ])
        
        # 创建证书
        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.utcnow()
        ).not_valid_after(
            datetime.utcnow() + timedelta(days=365)
        ).add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.DNSName("127.0.0.1"),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
                x509.IPAddress(ipaddress.IPv6Address("::1")),
            ]),
            critical=False,
        ).sign(private_key, hashes.SHA256())
        
        # 保存私钥
        with open(key_file, 'wb') as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))
        
        # 保存证书
        with open(cert_file, 'wb') as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
        
        print(f"SSL证书创建成功: {cert_file}, {key_file}")
        return True
        
    except ImportError:
        print("警告: cryptography库未安装，尝试使用OpenSSL命令...")
        return create_ssl_with_openssl(cert_file, key_file)
    except Exception as e:
        print(f"证书创建失败: {e}")
        return create_ssl_with_openssl(cert_file, key_file)


def create_ssl_with_openssl(cert_file='server.crt', key_file='server.key'):
    """使用OpenSSL命令创建SSL证书"""
    try:
        cmd = [
            'openssl', 'req', '-x509', '-newkey', 'rsa:2048',
            '-keyout', key_file, '-out', cert_file,
            '-days', '365', '-nodes',
            '-subj', '/C=CN/ST=Beijing/L=Beijing/O=Digital Human Assistant/CN=localhost'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"SSL证书创建成功: {cert_file}, {key_file}")
            return True
        else:
            print(f"OpenSSL命令执行失败: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("错误: 未找到OpenSSL命令，请安装OpenSSL或cryptography库")
        return False
    except Exception as e:
        print(f"OpenSSL证书创建失败: {e}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='数字人交互助手HTTPS服务器 - 通义千问版本')
    parser.add_argument('--host', default='localhost', help='服务器主机地址 (默认: localhost)')
    parser.add_argument('--port', type=int, default=8444, help='服务器端口 (默认: 8444)')
    parser.add_argument('--cert', default='server.crt', help='SSL证书文件路径 (默认: server.crt)')
    parser.add_argument('--key', default='server.key', help='SSL私钥文件路径 (默认: server.key)')
    parser.add_argument('--no-ssl', action='store_true', help='禁用SSL，使用HTTP协议')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("数字人交互助手HTTPS服务器 - 通义千问版本")
    print("=" * 60)
    
    # 检查通义千问API密钥
    api_key = os.getenv('DASHSCOPE_API_KEY')
    if api_key:
        print(f"✓ 通义千问API密钥已配置 (长度: {len(api_key)})")
    else:
        print("⚠ 警告: 未设置DASHSCOPE_API_KEY环境变量")
        print("  请设置您的通义千问API密钥:")
        print("  export DASHSCOPE_API_KEY='your-api-key'")
        print()
    
    # 检查并创建SSL证书
    if not args.no_ssl:
        if not create_ssl_certificate(args.cert, args.key):
            print("SSL证书创建失败，将使用HTTP协议")
            args.no_ssl = True
    
    # 创建HTTP服务器
    try:
        server = HTTPServer((args.host, args.port), DigitalHumanHandler)
        
        if not args.no_ssl:
            # 配置SSL
            context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            context.load_cert_chain(args.cert, args.key)
            server.socket = context.wrap_socket(server.socket, server_side=True)
            protocol = "HTTPS"
            url = f"https://{args.host}:{args.port}"
        else:
            protocol = "HTTP"
            url = f"http://{args.host}:{args.port}"
        
        print(f"\n🚀 服务器启动成功!")
        print(f"   协议: {protocol}")
        print(f"   地址: {url}")
        print(f"   时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n💡 访问说明:")
        print(f"   - 主页: {url}")
        print(f"   - API状态: {url}/api/status")
        print(f"   - 通义千问API: {url}/api/qwen")
        print("\n📝 使用提示:")
        print("   - 按 Ctrl+C 停止服务器")
        if not args.no_ssl:
            print("   - 首次访问可能需要信任自签名证书")
        print("=" * 60)
        print()
        
        # 启动服务器
        server.serve_forever()
        
    except KeyboardInterrupt:
        print("\n\n服务器已停止")
    except Exception as e:
        print(f"\n服务器启动失败: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()